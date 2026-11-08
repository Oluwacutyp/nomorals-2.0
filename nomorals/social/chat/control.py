"""Control commands: the owner's hands on the machine, from inside a chat.

While `nm chat` is running, the owner types slash-commands — from the local
console, or from the primary partner chat on any platform (a message that
starts with ``/`` and comes from the owner is a command, not conversation).

Commands:

    /help                        what you can do
    /status                      mood, relationship, platforms, autonomy
    /platforms                   per-platform health
    /start <platform>            hot-start a platform in this session
    /stop  <platform>            stop a platform in this session
    /mood                        show the ten dimensions + label
    /mood tired                  force the label (sets its profile's values)
    /mood energy=20 frustration=70   force dimensions directly
    /mood reset                  back to the persona's baselines
    /mode  off|suggest|auto      switch the autonomy mode, live
    /agent adaptive|autonomous|planning   switch the agent modes (combinable, per chat)
    /model [provider [fallback]] switch the model live (no restart, persists)
    /say <platform:chat> text    send a message as her, through the gateway
    /proposals                   pending autonomous messages
    /approve <id>                release a proposal
    /deny    <id>                kill a proposal
    /stage                       show relationship stage + trust
    /stage <stage>               advance/regress explicitly
    /power on <key>              unlock power mode (owner key)
    /power on <id> <passphrase>  unlock power mode (owner seal)
    /power off|status            lock / report
    /search <query>              quick web research + cited summary
    /searchdeep <query>          deep research (power mode)
    /searchleads                 legit paid-task platform report
    /money [scan|list|new]       money-making opportunities hunter
    /searchhist [n]              recent research runs
    /book <topic> [chapters]     writes a real book → PDF, sends it when done
    /book status [slug]          book progress · /book list · /book build <slug>
    /features [name on|off]      feature toggles (arena, vision, search, …)
    /arena [status|run|topics|stream|export|approve|deny]
                                 self-improvement arena control
    /trial [start|save|send|list|rm]
                                 single-account trial credentials
    /devon [free text]           the autonomous dev agent: plans tools, runs
                                 them, and digests a plain-English answer
    /quit                        stop the runtime

Parsing is pure (no I/O, no dependencies on the runtime) so it is fully
testable; dispatch happens in :mod:`nomorals.agents.partner_runtime`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from .failures import record_failure

__all__ = ["CONTROL_COMMANDS", "COMMAND_DETAILS", "ControlCommand",
           "parse_control", "PLATFORMS", "help_text", "help_menu", "whatsapp_menu",
           "menu_sections",
           "detailed_help", "capability_brief",
           "list_catalog", "LIST_GROUPS", "LIST_ONELINERS", "GAME_COMMANDS",
           "GAME_START_COMMANDS",
           "PUBLIC_COMMANDS", "PUBLIC_GAME_COMMANDS", "caller_may_run",
           "public_menu_text",
           "MAX_STACK_DEPTH", "NON_STACKABLE_COMMANDS", "STACK_DENIAL_REPLY",
           "split_command_stack", "expand_prev_placeholders", "stack_header",
           "WHATSAPP_UNSUPPORTED_KINDS", "TELEGRAM_CLEANUP_KINDS"]

PLATFORMS = ("telegram", "discord", "whatsapp", "local")

#: Kinds WhatsApp cannot execute at all (owner ruling 2026-11-04:
#: a command the platform cannot run must not sit on that
#: platform's surface).  The mute family's handlers bottom out in
#: the adapter's admin_restrict / admin_unrestrict verbs, and
#: WhatsApp has NO per-person mute — the adapter refuses honestly
#: (the only silences are removing the member, or the whole group
#: set to admins-only; neither is a mute).  The cleanup trio is
#: Telegram-only machinery: it reads Telegram's member list and
#: deleted-account markers, which WhatsApp does not expose at
#: all.  So on WhatsApp these kinds are absent from the menu and
#: from group help, and typing one gets the platform truth at
#: dispatch instead of a failed execution
#: (runtime.handle_control).  Telegram executes them fine; the
#: global catalog is untouched.
WHATSAPP_UNSUPPORTED_KINDS: frozenset[str] = frozenset(
    {"mute", "unmute", "cleanup", "rmdeleted", "rminactive",
     "tag", "bansender", "unbansender", "reactwipe", "boosters"})

#: The Telegram-only member-cleanup kinds — on WhatsApp they get
#: their own dispatch truth line (see the constant above).
TELEGRAM_CLEANUP_KINDS: frozenset[str] = frozenset(
    {"cleanup", "rmdeleted", "rminactive"})

#: The §16 P5 kinds (member tags, sender-chat ban, reaction
#: wipe, boost intel): Telegram affordances with no WhatsApp
#: equivalent at all — WhatsApp has no member tags, no
#: channel-identity posting to ban, no reaction wipe, and no
#: boost system.  Same treatment as the cleanup trio: absent
#: from WhatsApp menus and group help; typed there, they get
#: the platform truth at dispatch (runtime.handle_control).
TELEGRAM_P5_KINDS: frozenset[str] = frozenset(
    {"tag", "bansender", "unbansender", "reactwipe", "boosters"})

#: Kinds whose GROUP replies are delivered as Telegram
#: ephemeral answers (§16: visible only to the asker and the
#: bot) on the telegram-bot surface when the group's p5
#: settings allow — personal state never needs a public
#: audience.  A failed ephemeral attempt falls back to DM,
#: then to a pointer line: it is never silently public.
EPHEMERAL_GROUP_KINDS: frozenset[str] = frozenset({"balance"})

#: wave 87: the commands that start a game directly.  The Social Operator
#: lets these through in EVERY chat (games are social); the Core Mind only
#: adds the natural-language triggers on top, and only in the owner's DMs.
GAME_COMMANDS = (
    # easy
    "wordchain", "hangman", "numberguess", "two_truths", "wyrr", "spy",
    "auction", "trivia",
    # easy (wave G1): engine games that predated direct commands
    "20q", "rps", "digits",
    # medium
    "mafia", "king", "story", "rpg", "shop", "duel", "case",
    # pvp + raid boss (arena multiplayer)
    "pvp", "raid",
    # ambitious
    "world", "escape", "political",
    # wild (wave 95)
    "poker", "ttt", "bulls", "craps", "memory", "mines", "wordle",
    # arcade (wave 97)
    "2048", "snake", "connect4", "battleship",
    # casino (wave 98)
    "blackjack", "roulette", "slots",
    # inbox (wave F2): async, one message per turn, no clock
    "gomoku", "reversi", "checkers",
    # puzzles (gamesbooks-1.0)
    "sudoku", "anagram", "cryptogram",
    # math class (procedural maths, adaptive tiers)
    "math",
    # arena gear (persistent equipment)
    "inventory", "equip", "unequip", "repair", "level",
    # battle skills (learnable martial arts)
    "skill",
    # earnable titles (arena flair)
    "title",
    # RPG attributes (strength/stamina/mana/intelligence)
    "stats",
    # daily hunt (double-XP arena challenge)
    "daily",
    # per-game mastery tiers (non-arena depth: ranks, unlocks)
    "mastery",
    # player-to-player gifting (coins, gear, items — with confirmation)
    "gift",
)

#: The GAME_COMMANDS kinds that :meth:`PartnerRuntime.handle_control`
#: dispatches through its generic game branch (``_control_game``) rather
#: than an explicit branch. The excluded kinds all have their own
#: explicit ``handle_control`` branch (math, inventory/equip/unequip/
#: repair, level, skill, title, stats, daily, mastery, gift). The
#: any-chat path in ``_on_message`` handles ALL of GAME_COMMANDS before
#: ``handle_control`` ever sees them; this set exists so
#: ``handle_control`` stays total for direct callers (stacks, console,
#: tour/heal probes) and so the tour can skip exactly these — starting
#: a game is interactive state, never a read-only probe.
GAME_START_COMMANDS: frozenset[str] = frozenset(
    k for k in GAME_COMMANDS
    if k not in {"math", "inventory", "equip", "unequip", "repair", "level",
                 "skill", "title", "stats", "daily", "mastery", "gift"}
)

#: The public BotFather command list — the ONLY commands a non-owner
#: may run on the public Telegram bot. Owner-only commands stay hidden
#: from the menu AND denied by :func:`caller_may_run`.
#:
#: ⚠️  TENTATIVE — the 32 game commands below are a RECONSTRUCTION
#: (easy 8 + G1 3 + medium 7 + ambitious 3 + wild 7 + arcade 4), NOT the
#: confirmed BotFather list. Verify against the list already set in
#: BotFather and trim this constant to match — it is one line to trim.
#: Excluded on purpose: pvp/raid (need live opponents), casino 3
#: (betting), inbox 3 (async), puzzles 3. Those still RUN for everyone
#: via the any-chat game dispatch (games are social); they are just not
#: on the public menu.
PUBLIC_GAME_COMMANDS: tuple[str, ...] = (
    # easy (8)
    "wordchain", "hangman", "numberguess", "two_truths", "wyrr", "spy",
    "auction", "trivia",
    # easy, wave G1 (3)
    "20q", "rps", "digits",
    # medium (7)
    "mafia", "king", "story", "rpg", "shop", "duel", "case",
    # ambitious (3)
    "world", "escape", "political",
    # wild, wave 95 (7)
    "poker", "ttt", "bulls", "craps", "memory", "mines", "wordle",
    # arcade, wave 97 (4)
    "2048", "snake", "connect4", "battleship",
)

#: Every kind a non-owner may run on the public bot: the 19 named public
#: commands + /game + the 32 public game commands. ``/menu`` and ``/list``
#: are NOT in here — bare /menu is a menu trigger, not a menu item.
PUBLIC_COMMANDS: frozenset[str] = frozenset([
    "help", "ping", "search", "news", "translate", "tl", "book", "music", "play", "video", "podcast",
    "tts", "stt", "look", "image", "lens", "decode", "dns", "whois",
    "cipher", "wisdom", "think",
    "rules", "member", "glog", "gsettings", "strikes",
    "recheck", "joinq", "console", "undonotice",
    "flag", "rep", "repclear", "repdismiss",
    "steal", "sticker", "convert", "gif", "emoji", "customemoji",
    "emojiart", "react",
    "awoo", "baka", "bite", "blush", "bonk", "bored",
    "bully", "cringe", "cry", "cuddle", "dance", "facepalm",
    "feed", "glomp", "handhold", "handshake", "happy", "highfive",
    "hug", "kiss", "laugh", "lick", "nom", "pat",
    "poke", "pout", "shrug", "slap", "sleep", "smile",
    "smug", "stare", "thumbsup", "tickle", "wave", "wink",
    "yeet",
    "joke", "dadjoke", "progjoke", "chuck", "fact", "fact2",
    "catfact", "numfact", "advice", "quote",
    "dog", "cat", "fox", "meme", "riddle", "8ball", "choose",
    "activity", "pokemon", "history", "nasa", "calc",
    "qcard", "canvasquote", "iqc", "wndwqute", "meme2",
    "fakemsg", "fakettc", "nokiamsg", "fakeff", "fakeffduo",
    "ffgirl",
    "removebg", "qr", "shorturl", "unshorten", "bible",
    "couplepp", "pp",
    "balance", "mine", "work", "pay", "gshop", "quests", "treasury",
    "deposit", "withdraw", "rob", "bail", "coinflip",
    "top", "tax", "grant",
    "gear", "heist", "shoot", "kill", "assassinate", "sabotage",
    "gang", "war", "police", "bounty", "claimdrop",
    "poll", "vote", "milestones", "contributions", "boards",
    "webshot",
    "anime", "manga", "anichar", "charpic", "staff", "studio",
    "episodes", "animechars", "trailer", "streaming", "animereview",
    "genres", "genre", "topanime", "topmanga", "popular", "trending",
    "airing", "upcoming", "season", "topchars", "anischedule",
    "nextep", "recommend", "randomanime", "randommanga", "randomchar",
    "animequote", "favorite", "whatanime",
    "waifu", "neko", "kitsune", "husbando",
    "property", "buyproperty", "sellproperty", "rent",
    "listproperty", "auctions", "mortgage", "insure",
    "drop", "claim",
    "mute", "unmute", "ban", "unban", "kick", "warn", "pin",
    "unpin", "purge", "report",
    "cleanup", "rmdeleted", "rminactive",
    "tag", "bansender", "unbansender", "reactwipe", "boosters",
    "game",
    *PUBLIC_GAME_COMMANDS,
])


#: Group governance decisions: kinds that decide a treasury proposal
#: the bot itself opened in a group (the weekly gift proposal, a
#: threshold spend, a treasury poll's execution). They are NOT public
#: commands — in a DM they stay owner-only — but inside a group chat
#: they belong to the group's admins, and the runtime passes them to
#: their handlers there (the handlers enforce the admin rule). The
#: standing rule this set implements (owner report 2026-11-07): a
#: system message never instructs an action its own surface refuses —
#: a proposal posted in a room is decidable in that room.
GROUP_DECISION_COMMANDS: frozenset[str] = frozenset([
    "approve", "deny",
])


#: Commands that require an explicit confirmation before they run.
#: The gate lives in ``PartnerRuntime.handle_control``: a bare
#: ``/<kind> ...`` returns a confirm prompt (with Confirm/Cancel inline
#: buttons on the Bot API adapter); ``/<kind> confirm`` runs the stored
#: original request; ``/<kind> cancel`` drops it. Pending confirmations
#: expire after :data:`CONFIRM_TTL_S`. Confirmable implies owner-only —
#: a non-owner gets the grant deny, never a confirm prompt. ``/game
#: delete`` keeps its own bespoke confirm; ``trial rm`` is untouched.
CONFIRMABLE_COMMANDS: frozenset[str] = frozenset({"quit", "forget"})
#: Pending-confirmation lifetime, seconds (matches /game delete).
CONFIRM_TTL_S = 300.0


def public_menu_text() -> str:
    """Menu text for bare /start and bare /menu — the PUBLIC list only.

    Short: title + grouped command names + the tap-a-button line. No
    per-command descriptions (those live in /help); the keyboard
    carries the actions. Owner-only commands are never listed here —
    the owner reaches the full catalog via /list.
    """
    games = " ".join(f"/{name}" for name in PUBLIC_GAME_COMMANDS)
    return (
        "🥷 Devon — tap a button below, or type a command:\n"
        "\n"
        "talk — /help /ping /search /news /translate /tl /book\n"
        "make — /music /play /video /podcast\n"
        "see & hear — /tts /stt /look /image /lens\n"
        "tools — /decode /dns /whois /cipher /wisdom /think\n"
        "express — /steal /sticker /convert /gif /emoji "
        "/customemoji /emojiart /react\n"
        "groups — /rules /member /glog /gsettings /strikes "
        "/recheck /joinq /console /undonotice /flag /rep "
        "/repclear /repdismiss /report /mute /unmute /ban "
        "/unban /kick /warn /pin /unpin /purge /cleanup "
        "/rmdeleted /rminactive\n"
        "group purse — /balance /mine /work /pay /gshop "
        "/quests /treasury /deposit /withdraw /rob /bail "
        "/coinflip /top /tax /grant /property /buyproperty "
        "/sellproperty /rent /listproperty /auctions /mortgage "
        "/insure /drop /claim\n"
        "conflict — /gear /heist /shoot /kill /assassinate "
        "/sabotage /gang /war /police /bounty /claimdrop\n"
        "group life — /poll /vote /milestones /contributions "
        "/boards /webshot\n"
        f"play — /game {games}\n"
        "\n"
        "tap a button below to run it · /help for what each one does"
    )


def caller_may_run(kind: str, is_owner: bool) -> bool:
    """Capability grant: may this caller run this command kind?

    Owner → anything registered. Non-owner → the public list, or any
    game command (games are social — the menu curates the 52, the game
    dispatch stays open per the existing social contract). Everything
    else is owner-only: deny without grant.
    """
    if is_owner:
        return True
    return kind in PUBLIC_COMMANDS or kind in GAME_COMMANDS


#: kind -> (min_args, max_args) — used by /help and by validation.
CONTROL_COMMANDS: dict[str, tuple[int, int]] = {
    "help": (0, 1),
    "status": (0, 0),
    "ping": (0, 0),
    "platforms": (0, 0),
    "start": (1, 1),
    "stop": (1, 1),
    "mood": (0, 10),  # label, or several dim=NN pairs
    "mode": (1, 1),
    "agent": (0, None),  # /agent [adaptive|autonomous|planning|+x|-x|status|off|tick]
    "chatmode": (0, None),  # /chatmode [mode|status|list|feature <f> <on|off>] [target] — per-chat mode + feature toggles; the target is a chat NAME (any length), a phone number, or a full key, so the tail is unbounded
    "style": (0, None),  # /style | /style <field> <value> [target] | /style reset [target] | /style list — inspect + tune the mined style profile (owner-only); the target is a chat NAME (any length), a phone number, or a full key, so the tail is unbounded
    "doctor": (0, 0),  # /doctor — system health: registry + config + stores + adapters + providers (owner-only)
    "model": (0, 6),  # provider + up to 5 fallback providers
    "providers": (0, 0),  # /providers — probe the whole chain, live or why not
    # /say is ONE command with three forms (single registry entry — a
    # duplicate key here would silently discard one form): gateway send
    # (/say platform:chat text), named-voice TTS (/say <voice> <text>),
    # rich voice (/say [direction] [accent=…] <text>). The dispatch
    # branch in handle_control routes on the tail shape.
    "say": (1, None),
    "proposals": (0, 0),
    "approve": (0, 1),     # context kind: proposal approvals in DM (arg required there); in a group it executes treasury proposals, bare lists them
    "deny": (1, 2),       # /deny <id> [reason] — reason required for upgrade tickets
    "stage": (0, 1),
    "power": (1, 3),  # /power on <key> | /power on <identity> <passphrase> (seal)
    "quit": (0, 1),        # /quit [confirm|cancel] — the 1 arg is the confirm verb
    "profile": (0, 0),      # /profile — the profile-aware runtime (env + tuned knobs)
    # self-update from chat (Step 9): drain → checkpoint → restart → resume.
    # Owner-only: absent from PUBLIC_COMMANDS, so caller_may_run denies
    # non-owners in the policy layer (menu hiding is not the enforcement).
    "update": (0, 0),      # /update — pull + migrate + health-check + restart
    "pull": (0, 0),        # /pull — git pull --ff-only only, no restart
    "deploy": (0, 0),      # /deploy — update + restart (deploy this code)
    "restart": (0, 0),     # /restart — checkpoint missions + restart, no pull
    # search engine (Telegram / WhatsApp / console — same commands everywhere)
    "search": (1, None),     # /search <query> — free text, no length cap
    "searchdeep": (1, None), # /searchdeep <query>  (power mode)
    "searchleads": (0, 0),   # legit paid-task platform report
    "money": (0, None),      # money opportunities hunter: scan|list|new|profile
    "searchhist": (0, 1),    # /searchhist [n]
    # BookForge: write a real book → PDF → send when done
    "book": (0, None),       # /book <topic> [chapters] | status | list | build <slug> | send <slug> <p> <c>
    "books": (0, None),      # alias of /book — /books read <url|slug> | continue <slug> [n] | sources
    "novel": (0, None),      # /novel follow <url|title> | list | read <slug> [ch] | continue <slug> [n]
    # WisdomKeeper: esoteric corpus Q&A, history timeline, guided practice
    "wisdom": (0, None),     # /wisdom ask <q> | practice list|<id>|stop | timeline [t] | compare <t> | status
    "wis": (0, None),        # alias of /wisdom
    # Universal Decoder: identify + decode anything, send binary results
    "decode": (0, None),     # /decode <data> | /decode file:<path> | /decode hash <digest>
    # Cookie analysis & handling (the CookieLab)
    "cookielab": (1, None),  # /cookielab <header> | /cookielab file:<path> | /cookielab ingest <header>
    # Prompt/mission structuring sub-agent
    "structure": (1, None),  # /structure <objective> — the structured brief
    # one-pass investigation: decode → crack → OSINT → knowledge graph
    "investigate": (1, None),  # /investigate <hash|jwt|cookie|url|blob|file> [file]
    "monitor": (0, None),    # /monitor [add <target> [every Ns] | list | tick | rm <ref>]
    # real crypto: AES-256 sealed blobs + classic ciphers
    "cipher": (0, None),     # /cipher enc|dec … with <pass> · /cipher vault put|get|list|rm
    # feature flags + arena + trial accounts
    "features": (0, 2),      # /features | /features <name> on|off
    "arena": (0, None),      # /arena [status|run [topic]|topics|stream [n]|export [n]|approve <id>|deny <id>]
    "trial": (0, 5),         # /trial [list|start <p>|assist <p> [--yes]|signup <p>|confirm <tok>|status|sms [country]|sms code|inbox <service>|save <p> <login> <pass>|send <p>|rm <p>]
    "identity": (0, 3),      # /identity [show|set <field> <value>|clear] — the profile bank for signups
    # reachability wave: capabilities that existed with no chat surface.
    # Temp inboxes are per-user (chat_key + sender id); one user can
    # never read another's. All ten are owner-only (not PUBLIC_COMMANDS)
    # and off the BotFather menu (100-cap) — registered, runnable, /list-ed.
    "tempmail": (0, 1),      # /tempmail new | status | forget — bare shows the stored inbox
    "checkmail": (0, 2),     # /checkmail [read <n> | code | forget] — read your temp inbox
    "vnumber": (0, 2),       # /vnumber [new [country] | status | forget] — temp/SMS number
    "checksms": (0, 1),      # /checksms [code | forget] — read SMS on your temp number
    "browser": (1, None),    # /browser open|text|links|forms|headings <url> | takeover [url] | run <url> [goal] (owner-only)
    "paper": (0, None),      # /paper start|status|stop|list — simulated trading, never real money
    "people": (0, None),     # /people list|show|add|note — the people spine
    "commitments": (0, None),  # /commitments list|add|kept|broken|cancel|overdue|due|reliability|verify
    "decisions": (0, None),  # /decisions list|add <decision> [| chosen | prediction | conf%]|resolve <id> [good|bad] <outcome>|review|calibration
    "iddocs": (0, None),     # /iddocs list|add <type> <expiry YYYY-MM-DD> <vault:ref> [country]|expiring [days]|overdue|renew <id> (owner-only)
    "focus": (0, None),      # /focus start|pause|resume|stop|status|today|week|month
    "backup": (0, 1),        # /backup status | verify — read-only backup health (runs need the vault, stay scheduled)
    # expansion wave
    "game": (0, 12),         # /game [list|<name>|quit|leaderboard|stats|shop|balance]
    # math class — standalone game command (wired handler, not engine game)
    "math": (0, None),       # /math [tier|answer <a>|leaderboard|streak|daily|duel ...|set|teach ...|upvote ...|balance]
    "npc": (0, None),        # /npc list | talk <name> <msg> | mood <name> — the living cast
    "charagent": (0, None),  # /charagent list | create | talk | play | forget — persistent agents
    "dm": (0, 2),            # /dm mood [mood] — the narrator's persona
    # wave 87: direct game-start commands — games are social, so they work in
    # EVERY chat. In non-owner chats these commands are the ONLY game trigger;
    # natural language never launches a game there. ("arena" stays the
    # self-improvement arena — that game is /game arena.)
    # arena gear: persistent equipment with durability, grades, sets
    "inventory": (0, 0),   # /inventory — equipped gear, closet, consumables
    "equip": (0, 1),       # /equip [gear] — wear it into battle
    "unequip": (0, 1),     # /unequip [slot] — take gear off
    "repair": (0, 1),      # /repair [gear] — restore durability for coins
    "level": (0, 0),       # /level — XP, level, stat growth
    "skill": (0, 3),       # /skill [learn|upgrade <name>] — battle skills
    "title": (0, 3),       # /title [set <name>] — earnable titles
    "stats": (0, 2),       # /stats [<attr> [points]] — RPG attributes
    "daily": (0, 0),       # /daily — today's double-XP hunt
    "mastery": (0, 1),     # /mastery [game] — per-game mastery tiers
    "gift": (0, 4),         # /gift @name <coins|gear|item> — gifting (2-step confirm)
    # easy
    "wordchain": (0, 0), "hangman": (0, 1), "numberguess": (0, 0),
    "two_truths": (0, 0), "wyrr": (0, 0), "spy": (0, 0), "auction": (0, 0),
    "trivia": (0, 0),
    # easy (wave G1): 20 questions / rock-paper-scissors / digit memory —
    # engine games, now directly startable like every other game. "/2048"
    # (wave 97) already proved digit-leading command names parse and
    # route: parse_control is a plain dict lookup, and Telegram bot
    # commands may contain digits.
    "20q": (0, 0), "rps": (0, 0), "digits": (0, 0),
    # medium
    "mafia": (0, 0), "king": (0, 0), "story": (0, 0),
    "rpg": (0, 0), "shop": (0, 0), "duel": (0, 0), "case": (0, 0),
    # pvp + raid boss (arena multiplayer)
    "pvp": (0, 1), "raid": (0, 0),
    # ambitious
    "world": (0, 0), "escape": (0, 0), "political": (0, 0),
    # wild (wave 95)
    "poker": (0, 0), "ttt": (0, 0), "bulls": (0, 0), "craps": (0, 0),
    "memory": (0, 0), "mines": (0, 0), "wordle": (0, 0),
    # arcade (wave 97)
    "2048": (0, 0), "snake": (0, 0), "connect4": (0, 0), "battleship": (0, 0),
    # casino (wave 98)
    "blackjack": (0, 1), "roulette": (0, 0), "slots": (0, 1),  # optional bet arg — /slots <amount> quick-starts (bet + first spin in one command); /blackjack <amount> deals at that stake
    # inbox (wave F2): async, one message per turn, no clock
    "gomoku": (0, 0), "reversi": (0, 0), "checkers": (0, 0),
    # puzzles (gamesbooks-1.0)
    "sudoku": (0, 0), "anagram": (0, 0), "cryptogram": (0, 0),
    # wave 87: the Core Mind — manual override over natural-language routing
    "mind": (0, None),       # /mind [status|clear|<goal>] — the core mind
    "news": (0, 2),          # /news [run|briefing|status]
    "translate": (0, None),   # /translate [to <lang>] <text> | <url> | (as a reply) | lang <code> | status
    "tl": (0, None),          # alias of /translate
    "steal": (0, None),       # /steal [tags] — claim replied media into her library (§21)
    "sticker": (0, None),     # /sticker [text] — the maker: text/emoji, or replied image/video
    "convert": (0, None),     # /convert <name> — sticker/gif/mp4/mp3 conversions (§21); in a group, an amount tail is the §25 currency exchange (context kind)
    "gif": (0, None),         # /gif <what> — a GIF from her library / providers
    "emoji": (0, None),       # /emoji <words|emoji> — the full Unicode table, searched; /emoji make <words> creates one it lacks (§22)
    "customemoji": (0, None), # /customemoji list|add|ref|show|remove — her own set (§22)
    "emojiart": (0, None),    # /emojiart dot|ascii|mosaic|draw|combo|fancy|remix|save — text-native art (§22)
    "react": (0, None),       # /react [emoji] — react to the replied message (§22)
    # per-emotion reaction GIFs (wave C): /<emotion> [@someone] | as a reply —
    # sources + captions live in expression.REACTION_GIFS
    "awoo": (0, None), "baka": (0, None), "bite": (0, None), "blush": (0, None),
    "bonk": (0, None), "bored": (0, None), "bully": (0, None), "cringe": (0, None),
    "cry": (0, None), "cuddle": (0, None), "dance": (0, None), "facepalm": (0, None),
    "feed": (0, None), "glomp": (0, None), "handhold": (0, None), "handshake": (0, None),
    "happy": (0, None), "highfive": (0, None), "hug": (0, None), "kiss": (0, None),
    "laugh": (0, None), "lick": (0, None), "nom": (0, None), "pat": (0, None),
    "poke": (0, None), "pout": (0, None), "shrug": (0, None), "slap": (0, None),
    "sleep": (0, None), "smile": (0, None), "smug": (0, None), "stare": (0, None),
    "thumbsup": (0, None), "tickle": (0, None), "wave": (0, None), "wink": (0, None),
    "yeet": (0, None),
    # fun one-shots (FEI gap wave D): sources + chains live in
    # social/chat/fun.py — text from health-checked seed chains,
    # pictures attached natively, /calc on the games evaluator.
    # /animequote is NOT here: the anime wave (B) owns that name.
    "joke": (0, None), "dadjoke": (0, None), "progjoke": (0, None),
    "chuck": (0, None), "fact": (0, None), "fact2": (0, None),
    "catfact": (0, None), "numfact": (0, None), "advice": (0, None),
    "quote": (0, None),
    "dog": (0, None), "cat": (0, None), "fox": (0, None),
    "meme": (0, None), "riddle": (0, None), "8ball": (0, None),
    "choose": (1, None),      # /choose <a> or <b> [or <c>…]
    "activity": (0, None), "pokemon": (0, None),
    "history": (0, None), "nasa": (0, None),
    "calc": (1, None),        # /calc <expression> — games evaluator
    # makers & tools (FEI gap wave E): renderers, rungs and
    # chains live in social/chat/makers.py.  /quote is wave
    # D's text quote — NOT taken; the card maker is /qcard,
    # and the census skins are kinds over the one maker.
    "qcard": (0, None),       # /qcard [style] <name> | <text> (or reply)
    "canvasquote": (0, None),  # /qcard in its census skin
    "iqc": (0, None),         # /qcard skin
    "wndwqute": (0, None),    # /qcard skin
    "meme2": (0, None),       # /qcard, meme style
    "fakemsg": (0, None),     # /fakemsg <style> <name> | <text> (or reply)
    "fakettc": (0, None),     # /fakemsg, ttc style
    "nokiamsg": (0, None),    # /fakemsg, chat style
    "fakeff": (0, None),      # /fakemsg, ff style
    "fakeffduo": (0, None),   # /fakemsg, duo style — two speakers
    "ffgirl": (0, None),      # /fakemsg, ff style
    "removebg": (0, None),    # /removebg as a reply to a photo
    "qr": (0, None),          # /qr <text> · /qr read (reply)
    "shorturl": (1, None),    # /shorturl <url>
    "unshorten": (1, None),   # /unshorten <url>
    "bible": (1, None),       # /bible <book> <ch>[:<v>] · /bible search …
    "couplepp": (0, None),    # /couplepp — one artwork, a matched pair
    "pp": (0, None),          # alias of /couplepp
    # chat & privacy (FEI gap wave F): OWNER-ONLY account ops
    # on his personal surfaces — deliberately NOT in
    # PUBLIC_COMMANDS, so caller_may_run admits the owner
    # alone.  Verbs + truth tables live in
    # social/chat/chatops.py and the handler in
    # agents/partner/runtime_chatops.py; the automatic halves
    # (statusreact/anticall/antitag sensing) are personal-organ
    # toggles in the presence pack's home (PERSONAL_TOGGLES).
    "archive": (0, None),     # /archive [chat name] — archive a chat
    "unarchive": (0, None),   # /unarchive [chat name]
    "mutechat": (0, None),    # /mutechat [chat name] — mute the CHAT
    "unmutechat": (0, None),  # /unmutechat [chat name]
    "readchat": (0, None),    # /readchat [chat name] — mark read
    "unreadchat": (0, None),  # /unreadchat [chat name] — mark unread
    "star": (0, 0),           # /star as a reply — star that message
    "unstar": (0, 0),         # /unstar as a reply
    "forward": (1, None),     # /forward <chat> as a reply — forward that message there
    "unsend": (0, 0),         # /unsend as a reply — delete own for all
    "edit": (1, None),        # /edit <new text> as a reply (alias)
    "editmsg": (1, None),     # /editmsg <new text> as a reply
    "antiedit": (0, 1),       # /antiedit [on|off] — edit recovery
    "block": (0, None),       # /block <number/@user> or reply in a DM
    "unblock": (0, None),     # /unblock <number/@user>
    "anticall": (0, 1),       # /anticall [on|off] — reject WA calls
    "disappear": (1, None),   # /disappear <off|24h|7d|90d> [chat]
    "statusreact": (0, None),  # /statusreact [on|off|emoji <e>]
    "viewonce": (0, None),     # /viewonce [n] — reveal a captured view-once
    "vv": (0, None),           # /vv — short form of /viewonce
    "poststatus": (0, None),   # /poststatus [text] — post a status as the owner
    "chanstory": (1, None),    # /chanstory <channel> [text] — post to a channel as the owner
    "joinlink": (1, None),     # /joinlink <link or code> — join a WhatsApp group via its invite link
    "newgroup": (1, None),     # /newgroup <title> — create a Telegram group as the owner
    "newchannel": (1, None),   # /newchannel <title> — create a Telegram channel as the owner
    "leave": (0, None),        # /leave [chat name] — leave a group or channel as the owner
    "banchecker": (0, None),  # /banchecker [number] — honest signals
    "gcstatus": (0, 0),       # /gcstatus — this group's real stats
    "statusinfo": (0, 0),     # /statusinfo — alias of /gcstatus
    "pollres": (0, 0),        # /pollres as a reply to a poll
    "tagall": (0, None),      # /tagall [text] — mention all (capped)
    "hidetag": (0, None),     # /hidetag [text] — mention all, hidden
    "antitag": (0, 1),        # /antitag [on|off] — mass-tag DM flags
    "pulse": (0, None),       # /pulse [status|config|run|catchup] [slot]
    # /research is ONE command (single registry entry — a duplicate key
    # here would silently discard one meaning): ad-hoc web research on
    # any topic, plus the always-on research loop (status/ideas/approve/
    # deny/history/from/ask/docs). Legal questions live under /legal.
    "research": (0, None),
    "code": (1, None),       # /code <what to build> — the coding bot
    "checkpoint": (0, None),  # /checkpoint [label] — snapshot the coding workdir
    "rewind": (0, 1),      # /rewind [n] — restore the nth-latest checkpoint
    "checkpoints": (0, 0),  # /checkpoints — list saved checkpoints
    "py": (1, None),         # /py <python code> — run in the sandbox (-s/-r sessions)
    "remember": (1, None),   # /remember <text> [kind] [tags:a,b]
    "review": (0, 1),       # /review [days] — "how did I do?" representation review
    "tutor": (0, None),       # /tutor [guide me] <topic> | answer <text> | hint | stop | status
    "learn": (0, None),        # /learn tutor|quiz|flashcards|mastery|waec|jamb — one front door over learn/
    "immune": (0, None),      # /immune status|scan|antibodies|quarantine|disarm|arrest|fever|drift|casefile — defensive immune system (owner-only)
    "impossible": (0, None),  # /impossible solve <problem> | status | attempt | dossier | queue | charter | lattice | examples — always-on conceive-and-solve (owner-only)
    "tour": (0, 1),         # /tour — smoke-test every command; /tour <cmd> deep-probes one
    "heal": (0, 1),         # /heal [--dry] | /heal <cmd> — self-healing loop: diagnose + auto-fix broken commands
    "ekiti": (0, None),       # /ekiti converse <text> | debate <topic> | say <text> | check "<expected>" | tones
    "course": (0, None),      # /course <topic> | waec <subject> | set <field> <value> | build | list | teach <id> | status
    "health": (0, None),      # /health log <text> | timeline | summary [days]
    "train": (0, None),       # /train plan <goal> [equipment] [weeks] | today | readiness | log [completed|skipped] [rpe] | list
    "form": (0, None),        # /form analyze <video> <exercise> | gait <video> | movements | apply <id>
    "film": (0, None),        # /film analyze <video> <game> [exercise] | games | fingerprint [game] | list [game] | export <id>
    "audio": (0, None),       # /audio edit <file> [lang] | fillers <file> [lang] | enhance <file>
    "overview": (0, None),    # /overview make <format> [lang] <Title :: text ;; ...> | list | ask <q> | voices
    "character": (0, None),   # /character talk <name> <question> | list | add <name> [book] | cutoff <name> <N>
    "audiobook": (0, None),  # /audiobook make <epub> [voices...] [stores...] | status
    "graph": (0, None),       # /graph add <type> <label> | link <from> <to> <type> | disrupt <id> [note] | show [id] | breaks <id> | path | sync
    "route": (0, None),       # /route plan <s1>; <s2>; ... | add <label> [lat,lng] | stops | record <from> > <to> <min> [kobo] | stats
    "eta": (0, None),         # /eta <task> [seg=min] ... | record <task> <pred> <actual> | risk <task> <elapsed> | stats [task]
    "congestion": (0, None),  # /congestion status | register <name> [kind] [capacity] | advise <resource> [agents=N] | predict <resource> | alternate <r> <alt>
    "scamcheck": (0, None),   # /scamcheck <jiji/propertypro link or listing text> [photo:<path>] — six automated scam checks
    "passport": (0, None),    # /passport generate | show | income <band> | history/ref/doc add — verify-once rental passport (owner only)
    "truecost": (0, None),    # /truecost <listing text> | afford <listing> — real move-in cost (owner only)
    "value": (0, None),       # /value <desc> — DIY valuation: range + confidence band + comparables
    "valuepick": (0, None),   # /valuepick <estimate id> <1,2,4> — refine from picked comparables
    "watch": (0, None),       # /watch 2bed Yaba under 1.5m [for 7|14|28d] | list | stop <id> — persistent match alerts
    "aeo": (0, None),         # /aeo track <brand> [prompts] | report [brand] | briefs [brand] — share-of-answer visibility (owner only)
    "send": (0, None),        # /send campaign <template> to <a,b> [via <provider>] | queue | process | status — self-hosted send layer (owner only)
    "leads": (0, None),       # /leads trigger <post> <keyword> | list | dm <contact> | everyone <post> — comment→DM→lead loop (owner only)
    "brief": (0, None),       # /brief <topic> | run <topic> — SERP-derived content brief (owner only)
    "content": (0, None),     # /content score <draft> | schedule <id> <when> | run <topic> — brief-first pipeline (owner only)
    "guardrails": (0, None),   # /guardrails add <rule> | list | run | override … — rule-based ad-spend safety (owner only)
    "cost": (0, 2),          # /cost [on|off] | /cost budget $0.50 — transparent spend display (owner only)
    "competitor": (0, None),   # /competitor track <account> | report <account> | digest | vs <them> <you> — no-access competitor intel (owner only)
    "routine": (0, None),     # /routine <natural language> | confirm <id> | list
    "home": (0, None),        # /home status | what changed | unusual?
    "store": (0, None),       # /store provision <biz> | woo <biz> <url> | list | catalog <id> <desc> | <id> <instruction>
    "mandate": (0, None),     # /mandate issue <scope> <per-txn> [per-day] [days] | list | revoke <id> | stop all
    "track": (0, None),       # /track LOS LHR 2026-12-01 [under 400k] | list
    "untrack": (1, None),     # /untrack <watch_id>
    "trip": (0, None),        # /trip | /trip <id> | /trip add <text> | /trip calendar <id> | /trip docs <id>
    "travelclient": (0, None),  # /travelclient create <name> | knowledge <id> <title> | <text> | ask <id> <q> | spend <id> | list | viki [airline]
    "recall": (0, None),     # /recall [query] — what she has stored
    "forget": (1, None),     # /forget <id or description>
    "memories": (0, 1),      # /memories [name] — trust view: what she remembers
    # expense tracking + conversational budgeting (Naira-first)
    "spend": (1, None),      # /spend <amount> [on <category>] [note...]
    "budget": (1, None),     # /budget set <category> <amount> | /budget list
    "spending": (0, 1),      # /spending [week|month] — summary vs budgets
    "miniapp": (0, None),      # /miniapp … — group mini-apps (poll/expenses/rsvp)
    "panel": (0, None),        # /panel … — persistent live-data panels (Hark pattern)
    "gtrip": (0, None),        # /gtrip … — group-travel stack (polls, expenses, legs)
    "contract": (0, None),     # /contract review [type] <text> | types — letter-grade contract risk review (Nigerian playbooks)
    "legal": (0, None),        # /legal [language] <question> — plain-language legal information (EN/PCM/YO/HA/IG)
    "contracts": (0, None),    # /contracts … — contract portfolio: obligations, renewals, SLA tracking
    "regwatch": (0, None),     # /regwatch … — regulatory change monitoring (CBN/SEC/NDPA/FIRS)
    "uprofile": (0, None),     # /uprofile … — prompt-scaffolded profiles + element-level likes (gig/community/business)
    "vnote": (0, None),        # /vnote … — voice notes in chat (audio + searchable transcripts)
    "match": (0, None),        # /match … — curated daily batches + stable two-sided matching
    "cgroup": (0, None),       # /cgroup … — themed groups, events, meetups (#84)
    "challenge": (0, None),    # /challenge … — photo-proof challenges + streaks + squads (#87)
    "room": (0, None),           # /room … — live audio rooms: host → speakers → hand-raise (#107)
    "party": (0, None),          # /party … — group parties: DJ queue, votes, trivia, wall, recap (AA-05)
    "ajo": (0, None),            # /ajo … — rotating savings circles (AA-22)
    # new layer: voice, scheduler, db, api, vision, swarm
    "tts": (1, None),        # /tts <text> — speak it (sends a voice note; file=on sends the file)
    "stt": (1, 5),           # /stt <path> — transcribe an audio file
    "sing": (1, None),       # /sing <melody> — sing it (C4:0.5:hello …)
    # (no second "say" entry: the single entry above covers all three forms)
    "look": (1, None),       # /look <path|url> [focus] — screen-reader analysis
    "schedule": (0, None),   # /schedule add|list|rm|enable|disable|run|status
    "db": (0, None),         # /db tables | schema <t> | query <sql> | counts
    "artifact": (0, None),   # /artifact list | info <ref> | send <ref> — saved artifacts, sent to this chat
    "api": (0, None),        # /api list | <connector> [json params]
    "connectors": (0, 2),   # /connectors [list|status|connect] [name] — manage the service connectors
    "provision": (0, None),  # /provision [service] | login <service> <username> — Devon fetches missing API keys herself (owner only)
    "pair": (0, None),        # /pair [whatsapp <number> [label]|code|start|stop|restart|remove <id>] — pairing-with-code account pool (§15; owner only)
    "massreact": (0, None),   # /massreact <post link> | <emoji> [| count] [| live] — react from every paired account (§14; owner only, arm-gated)
    "pollvote": (0, None),    # /pollvote <poll link> | <option> [| count] [| live] — vote from every paired account (§14; owner only, arm-gated)
    "chfollow": (0, None),    # /chfollow <channel link> [| live] — every paired account follows (§14; owner only, arm-gated)
    "rules": (0, 0),          # /rules — this group's rules, rendered from the settings the enforcer reads (P1)
    "member": (0, 1),         # /member [@name] — group member card: tenure, activity, identity changes (P1)
    "glog": (0, 1),           # /glog [n] — the group's held audit log (admins; P1)
    "strikes": (0, 1),        # /strikes [@member] — the defense strike ledger: who's on the ladder (admins; §17)
    "recheck": (0, 0),        # /recheck — a restricted member's “check me again”: re-runs the join criteria live (§28)
    "joinq": (0, None),       # /joinq [approve|decline <id>|all] — the admin join queue: requests + raid holds (§28)
    "console": (0, None),     # /console [gN [section [grammar]]] — the DM admin console: groups the sender admins, configured in tappable sections (§30)
    "undonotice": (0, None),  # /undonotice <id> — undo one automated action from its notice: strike removed, platform action reversed (§31)
    "flag": (0, None),        # /flag [@member <category> [note]|ok|no <id>|queue|retract @member] — cross-group reputation flags: staged, admin-confirmed (§32)
    "rep": (0, None),         # /rep [@member] — the reputation book: one person's file, or the book's shape (§32)
    "repclear": (0, None),    # /repclear <member id> — lift a reputation restriction in this group (§32)
    "repdismiss": (0, None),  # /repdismiss <record id> — dismiss a reputation record's warnings for this group (§32)
    "gsettings": (0, None),   # /gsettings [welcome|goodbye|clean on|off|rules add|clear|flood <n>|links ...] — group settings (admins; P1)
    "balance": (0, 0),        # /balance — your sparks + the group treasury (P2 economy)
    "mine": (0, 0),           # /mine — claim mined sparks at the decaying rate (P2)
    "work": (0, 0),           # /work — a cooldown-bound shift; pay follows engagement + level (P2)
    "pay": (0, None),         # /pay @member <amount> — transfer sparks; the tax % feeds the treasury (P2)
    "gshop": (0, None),       # /gshop [buy <item>] — the group status shop (P2; 'shop' is the shop GAME)
    "quests": (0, None),      # /quests [claim <id>] — group quests and their rewards (P2)
    "treasury": (0, 0),       # /treasury — the group's purse: balance, circulation, health (P2)
    "drop": (0, None),        # /drop <total> <per claim> — admin funds a giveaway pool (P2)
    "claim": (0, 0),          # /claim — take your share of the live drop (P2)
    "deposit": (0, 1),        # /deposit 50|all — wallet → bank vault, safe from robbery
    "withdraw": (0, 1),       # /withdraw 50|all — bank vault → wallet
    "rob": (0, 1),            # /rob @member (or reply) — steal wallet sparks; failure is jail
    "bail": (0, 0),           # /bail — pay the treasury to leave jail now (or pay a hospital bill)
    "gear": (0, None),        # /gear [buy|install|use <item> ...] — your conflict loadout; the arsenal shelf is in /gshop (§23 addendum)
    "heist": (0, None),       # /heist [start treasury|bank <district>|vault @member | join | go] — crew scores vs installed defenses (§23)
    "shoot": (0, 1),          # /shoot @member (or reply) — geared hit verb: small take, failure is jail (§23)
    "kill": (0, 1),           # /kill @member — needs a real weapon; bigger take + hospital for the target (§23)
    "assassinate": (0, 1),    # /assassinate @member — needs a heavy weapon; the biggest hit (§23)
    "sabotage": (0, 1),       # /sabotage @member — wreck their best property; the estate bills the damage (§23)
    "gang": (0, None),        # /gang [create <lifestyle> <name>|join|leave|job|deposit|withdraw|claim|tax|list] — gang life (§24); §36: register|chapter <name>|hq [claim|build|move]|vault|armory|code|redeem|side <gang>
    "war": (0, None),         # /war [declare <gang>|accept|decline|join|spoils <district>|raid <gang>] — gang wars, in-group + inter-group (§23); raid = the §36 HQ raid
    "police": (0, None),      # /police [bribe [envelope]] — the precinct board: wanted, bounties, your heat (§24)
    "bounty": (0, None),      # /bounty [post @member <n>|hunt @member] — heads and hunters (§24)
    "claimdrop": (0, 1),      # /claimdrop [ref] — claim a system drop (§35); claims are settlements
    "poll": (0, None),        # /poll — group polls; treasury polls execute through governance (P3)
    "vote": (0, 2),           # /vote [poll] <option> — one vote per member; buttons carry this command
    "milestones": (0, None),  # /milestones — the group's memory book (P3)
    "contributions": (0, None),  # /contributions [top|appeal N] — §26 transparency + the reviewed ranker
    "boards": (0, None),         # /boards [name] — the §14 board organ: scopes × metrics, movement vs the last publish
    "webshot": (0, None),        # /webshot [full] [phone|tablet|desktop] [dark] [idle] [pdf] [annotate] [clip <sel>] <url> — render a page, return the shot (§18/§20); watch rides it
    "owner": (0, None),          # /owner [status|audit|bind|unbind|sudo|deputy|mark|rebind] — the §19 owner system in DM (owner only; the record is never editable from a group)
    "sudo": (0, None),           # /sudo [add|remove] — the §19 sudo rung in plain words (owner only; edits delegate to the owner grammar)
    "coinflip": (0, 2),       # /coinflip 10 heads — casino sink at 1.9×
    "top": (0, 0),            # /top — the richest board (wallet + bank)
    "tax": (0, 0),            # /tax — the group's monetary policy, stated
    "grant": (0, None),       # /grant @member 100 — admin pays from the treasury (governance above threshold)
    "property": (0, 1),       # /property [manager] — districts + market index, your portfolio, your gems (§17/§20 real estate)
    "buyproperty": (0, None), # /buyproperty <district> <tier> | public <district> <tier> — buy a building from the system (admins: public stock via governance)
    "sellproperty": (1, 1),   # /sellproperty <id> — sell back at the depreciated rate
    "rent": (0, 0),           # /rent — collect accrued property rent, paid in gems 💎
    "listproperty": (0, None),# /listproperty [<id> <ask> | buy <listing> | cancel <id>] — the member property market (treasury cut on sales)
    "auctions": (0, None),    # /auctions [start <id> <start> [hours] | bid <auction> <amount>] — property auctions, treasury escrow, anti-snipe
    "mortgage": (0, None),    # /mortgage [borrow <id> <amount> | repay <id> <amount>] — treasury loans against property; bank-debited weekly; default = repossession
    "insure": (0, None),      # /insure [buy <id> | pay <id>] — 30-day cover against repair waves; rent offsets repair debt
    "mute": (0, None),        # /mute [@member] [10m|2h|1d|inf] — admin: reply to their message or name them (P1 §4)
    "unmute": (0, 1),         # /unmute [@member] — admin: lift a mute
    "ban": (0, None),         # /ban [@member] [reason] — admin: reply or name
    "unban": (0, 1),          # /unban [@member] — admin: lift a ban
    "kick": (0, None),        # /kick [@member] [reason] — admin: out now, may rejoin
    "warn": (0, None),        # /warn [@member] [reason] — admin: warns escalate to an auto-mute at the limit
    "pin": (0, 0),            # /pin — admin: pin the message you reply to
    "unpin": (0, 0),          # /unpin — admin: unpin the current pin
    "purge": (0, 0),          # /purge — admin: delete the message you reply to
    "report": (0, None),      # /report [@member] [reason] — member: flag something for the admins (held log)
    "cleanup": (0, 1),        # /cleanup [page] — admin: Telegram group cleanup SCAN (dry run, removes nothing): deleted accounts + members inactive past the window
    "rmdeleted": (0, 0),      # /rmdeleted — admin: remove the deleted accounts the scan finds (kick, not ban; journaled per member)
    "rminactive": (0, 1),     # /rminactive [days] — admin: remove members inactive past the window (setting, or days for this run)
    "tag": (0, None),         # /tag @member <words>|off — admin: set/clear a member's tag on the Telegram bot surface (§16 P5)
    "bansender": (0, None),   # /bansender — admin: ban the CHANNEL identity behind a replied-to channel post (§16 P5)
    "unbansender": (0, None), # /unbansender — admin: lift a sender-chat ban (§16 P5)
    "reactwipe": (0, None),   # /reactwipe @member — admin: wipe a member's recent reactions in this chat (§16 P5)
    "boosters": (0, 1),       # /boosters [@member] — the room's boost board, or one member's boosts (§16 P5)
    "notion": (0, None),      # /notion dbs|query|add — Notion reads + page writes
    "gcal": (0, None),        # /gcal [agenda] | add — Google Calendar
    "trello": (0, None),      # /trello boards|lists|cards|add — Trello
    "exness": (0, None),      # /exness balance|positions|price|buy|sell|close — Exness trading
    "stripe": (0, None),      # /stripe balance|customers|charges|link — Stripe payments
    "swarm": (1, None),      # /swarm <goal> [workers] — parallel devon agents
    # power layer: network, proxies, scripts, osint, macros
    "dns": (1, 5),           # /dns <domain> [record type]
    "scan": (1, 6),          # /scan <target> [ports] [banner] — a URL target routes to the security-audit exposure scan instead
    "audit": (0, None),      # /audit <repo url|path> — security audit of a repo (secrets incl. history, dependencies, insecure patterns) + disclosure draft; /audit own add|remove|list <asset> — the own-asset registry (owner only)
    "whois": (1, 3),         # /whois <domain>
    "ports": (0, 1),         # /ports — what is listening here
    "proxy": (0, 8),         # /proxy status|scrape|test|health|pool|pick|affinity|rotate on|set <url>|ssh <…>
    "toolhub": (0, 2),       # /toolhub | /toolhub proxy [run] | /toolhub bin [digits] | /toolhub help — button tool hub (owner-only; NOT /hub, which is the media hub)
    "workspace": (0, 4),     # /workspace [status|scale <n>|up|down|pause|resume|add|remove]
    "gen": (2, None),        # /gen <kind> <name> [json config]
    "osint": (1, None),      # /osint <target> | campaign <seeds> | graph <verb>…
    "record": (0, None),     # /record start <name>|stop|step <tool> [json]|status
    "macro": (0, None),      # /macro [list] | /macro <name> [json overrides]
    "file": (1, None),       # /file <platform> <chat> <path> [caption]
    "publish": (1, None),    # /publish <platform> <chat> <md path> [format]
    "deliver": (1, None),     # /deliver report <topic> [--section "T::body"] [--to p:c]
    "data": (0, None),       # /data mine [name] | /data list | /data fetch <ref> [rows]
    "evolve": (1, None),     # /evolve <instruction> | /evolve apply <id> | /evolve list
    "upgrade": (0, None),    # /upgrade list|show|diff|approve|deny|applied — research→approve→evolve review loop
    "speak": (1, None),      # /speak <text> — neural TTS voice note back in chat
    "voice": (0, None),      # /voice … — the voice catalogue (list/use/say/clone/…)
    "clonevoice": (1, 1),  # /clonevoice <name> — attach a voice note, clone it
    "voices": (0, 0),      # /voices — list every voice, cloned vs built-in
    "bet": (0, None),        # /bet analyze <home> vs <away> [odds…] | bankroll | backtest
    "desk": (0, None),       # /desk status|study|signals|library|collect|sources|provision — the betting desk (crash lab + signal calls)
    "apk": (0, None),        # /apk add|list|status|show|dissect|teach|diff|bisect|signatures|hexpatch|artifacts|send — ApkForge: APK intake, dissection, editing/building (A2), version bisect + signature growth + native patching (A3)
    "re": (0, None),         # /re status|inspect <path>|evidence [id]|unknowns|close <id> [observed|derived] <resolution>|audit|provision|graph|mission <q> -- <targets>|missions|mstatus|mresume — the RE organ: evidence-led static inspection (Evidence records + residual unknowns; spine + provisioned deep rungs + firmware/traffic + Pass 4 missions/reconstruction)
    "pump": (0, None),       # /pump status|watch|study|dossier <mint>|calls|wallet <addr>|graph|attention <mint>|learn|shelf <q>|sources|watch|provision — the pump.fun intelligence desk (signals only)
    "finance": (0, None),    # /finance quote|analyze|signal|idea|backtest|strategies|compare|watch|doctor
    "weather": (0, None),    # /weather [place] — live conditions + forecast + alerts
    "tz": (0, None),         # /tz [time] — timezone conversion, labeled in owner tz
    "task": (1, None),       # /task add <instruction>|run [id]|list
    "notify": (0, 1),        # /notify [n] — recent alerts
    "proactive": (0, 0),     # /proactive — push-send switches + delivery states
    "email": (0, None),        # /email triage|inbox|search|read|thread|drafts|compose|reply|send|approve|discard|followups — full mailbox (owner only)
    "gmail": (0, None),        # alias of /email
    "mail": (0, None),         # alias of /email
    "inbox": (0, None),        # alias of /email — bare /inbox lists the inbox
    "mission": (0, None),    # /mission status|list|stall|clear|pause|resume|cancel|retry|watch|unwatch|new
                             #   /mission new <research|build|fix> <args>
    "image": (1, None),      # /image <path-or-url> — lookup; <prompt> — generate
    "imggen": (0, None),     # /imggen <prompt> — Devon's own diffusion studio
    "shorts": (0, None),     # /shorts make|status|resume|niches|calendar|ledger|estimate
    "lens": (1, 3),          # /lens <path-or-url> — reverse image search
    # devon: the autonomous dev & investigation agent
    "devon": (0, None),      # /devon [free text] — plan tools, run, digest, reply
    # reasoning: explicit, auditable multi-step thought
    "think": (1, None),      # /think <question> [strategy] — show the work
    "benchmark": (0, 2),     # /benchmark [dimension] — how sharp is the system right now
    # wave 68: command discovery
    "list": (0, 2),          # /list [group] [page] — every executable command, categorized
    "commands": (0, 2),      # alias of /list
    # redteam: defensive self-attack suite
    "redteam": (0, 1),       # /redteam [scenario_id] — attack our own loop, report holes
    "balances": (0, None),   # /balances [rail…] — unified money view
    "alert": (0, None),       # /alert add|list|remove|check|snooze|sync-bills
    "goal": (0, None),        # /goal add|list|done|snooze|remove
    "insights": (0, None),    # /insights [days] — native money analytics
    "predict": (0, 3),       # /predict <stake> <over|under> <1-99> — prediction pit (progression-unlocked)
    "menu": (0, 2),          # alias of /list ([group] [page] ride the same grammar)
    # wave 72 systems: media, execution, archives, builders
    "music": (0, None),      # /music <topic> [style] | styles | song [slug] | bed | full | voices
    "distribute": (0, 1),   # /distribute [song.wav] | legal
    "play": (0, None),       # /play <paths…> | status | queue | pause | …
    "download": (1, None),   # /download <url> [audio] — any-URL media downloader
    "tiktok": (1, None),     # /tiktok <url> [audio] — TikTok downloader (watermark-honest routes)
    "ttdl": (1, None),       # alias of /tiktok
    "instagram": (1, None),  # /instagram <url> — public reel/post downloader (login wall named)
    "spotify": (1, None),    # /spotify <url> — identify on Spotify, fetch the full track elsewhere
    "aio": (1, None),        # /aio <url> [audio] — all-in-one: any supported link, routed by platform
    "video": (0, None),      # /video <query> | download <url> | platforms
    "director": (0, None),   # /director cover|plan|shoot|regen|show — coverage → design → shoot → QA
    "caption": (0, 1),       # /caption [style] — burn AI subtitles into a video (attach or path)
    "vision": (0, None),     # /vision [question] — analyze an image (attach or path/URL)
    "exec": (0, None),       # /exec <code> | languages — multi-language sandbox
    "zip": (0, None),        # /zip <paths…> --dest x.zip | list | info | extract | digest
    "apps": (0, None),       # /apps [list|build|serve|stop|served|stacks|info]
    # wave 73: media orchestrator + CI loop
    "hub": (0, None),        # /hub [song <topic…> [style]|video <q…>|podcast <q…>|status]
    "podcast": (1, None),    # /podcast <query…> — find→download→transcribe→chapters
    "dj": (0, 1),           # /dj [trending|<genre>] — radio show w/ voice breaks
    "produce": (0, None),   # /produce [spotify-url|vibe words…] — compose original music (taste-aware)
    "like": (0, None),       # /like [notes…] — last production was good
    "dislike": (0, None),    # /dislike [notes…] — last production missed
    "cookies": (0, 1),       # /cookies [check] — YouTube cookie file status/setup
    "sham": (0, 0),          # /sham — reply to audio/voice note to identify the song
    "musicnote": (1, 1),     # /musicnote find|create — answer the music voice-note ask (owner-only)
    # anime organ (FEI gap wave B): AniList-first records with a
    # Jikan second seed; handlers own their tail grammar, so every
    # kind takes the unbounded tail like /play and /music do.
    "anime": (0, None),      # /anime <title> | open <n> — anime lookup (AniList + MAL cross-check)
    "manga": (0, None),      # /manga <title> | open <n> — manga lookup
    "anichar": (0, None),    # /anichar <name> | open <n> — anime character lookup (FEI "character"; /character is Devon's story cast)
    "charpic": (0, None),    # /charpic — a random anime character picture
    "staff": (0, None),      # /staff <name> | open <n> — voice actors, directors, writers
    "studio": (0, None),     # /studio <name> | open <n> — studio record + best-known works
    "episodes": (0, None),   # /episodes <title> — episode list + guide
    "animechars": (0, None), # /animechars <title> — a title's cast
    "trailer": (0, None),    # /trailer <title> — its trailer (watch via /video)
    "streaming": (0, None),  # /streaming <title> — legal watch links on the record
    "animereview": (0, None),  # /animereview <title> — scores + community reviews (FEI "review"; /review is the representation review)
    "genres": (0, None),     # /genres — the genre list on file
    "genre": (0, None),      # /genre <name> — top anime in a genre
    "topanime": (0, None),   # /topanime — highest-scored anime
    "topmanga": (0, None),   # /topmanga — highest-scored manga
    "popular": (0, None),    # /popular [manga] — most popular right now
    "trending": (0, None),   # /trending [manga] — trending now
    "airing": (0, None),     # /airing — currently airing, with next-episode countdowns
    "upcoming": (0, None),   # /upcoming — announced, not yet out
    "season": (0, None),     # /season [winter|spring|summer|fall] [year] — a season's anime
    "topchars": (0, None),   # /topchars — most-loved characters
    "anischedule": (0, None),  # /anischedule — the week's airing grid in WAT (FEI "schedule"; /schedule is the cron surface)
    "nextep": (0, None),     # /nextep <title> — when its next episode airs (WAT)
    "recommend": (0, None),  # /recommend <title> — more like it, from the record
    "randomanime": (0, None),  # /randomanime — a random anime record
    "randommanga": (0, None),  # /randommanga — a random manga record
    "randomchar": (0, None),   # /randomchar — a random character record
    "animequote": (0, None),   # /animequote [title] — a quote, attributed
    "favorite": (0, None),     # /favorite [add <title>|remove <n>|open <n>] — YOUR saved list (Devon's store)
    "whatanime": (0, None),    # /whatanime — reply to a frame: which anime, episode, timestamp (trace.moe)
    "waifu": (0, None),      # /waifu — a waifu picture (pool APIs)
    "neko": (0, None),       # /neko — a neko picture
    "kitsune": (0, None),    # /kitsune — a kitsune picture
    "husbando": (0, None),   # /husbando — a husbando picture
    "fix": (1, None),        # /fix <code> [lang] [--rounds N] — run until the model gets it green
}

_HELP_TEXT = "\n".join(
    [
        "control commands (you, the owner — start a message with /):",
        "you don't need to memorize these — just say what you want in plain words:",
        "  “write me a book about X” → book   “play X” → play   “research Y” → search",
        "  “compose a song about Z” → music   “create a spotify account” → account",
        "  “build a todo app” → code   “let's play chess” → games",
        "  /status /platforms /help",
        "  /start telegram | /stop telegram        hot start/stop a platform",
        "  /mood [label | dim=0..100 …] | /mood reset",
        "  /mode off|suggest|auto                  autonomy, live",
        "  /profile [save|reload]                 this machine's runtime tune (wave 86)",
        "  /mind [status|<goal>]                  the core mind — route a goal, inspect the routing",
        "  /model [provider [fallback …]]          switch model live (persists)",
        "  /providers                          probe every provider: live or why not",
        "  /say telegram:123 text…                 send as her",
        "  /proposals /approve <id> /deny <id>",
        "  /stage [committed|dating|…]",
        "  /power on <key> | /power on <identity> <passphrase> | /power off | /power status",
        "  — search —",
        "  /search <query>                         quick research + cited summary",
        "  /searchdeep <query>                     deep research (power mode)",
        "  /searchleads                            legit paid-task platforms report",
        "  /money [scan|list|new]                   money-making opportunities hunter",
        "  /searchhist [n]                         recent research runs",
        "  — bookforge (writes a real book → pdf, sends it) —",
        "  /book <topic> [chapters]                start a book (auto-sends the pdf)",
        "  /book status [slug] | /book list        progress",
        "  /book build <slug> | /book send <slug> <p> <chat>",
        "  — story reader (webnovels: freewebnovel · novelfull · royalroad) —",
        "  /book search-stories <title>             find a story across archives",
        "  /book follow <url|title>                follow it (progress tracked)",
        "  /book read <slug> [n] | next | prev     read chapters",
        "  /book bible <slug>                      auto-built story bible",
        "  /book continue <slug> [n]               keep the story going in its voice",
        "  /books read <url|slug> [n] | continue <slug> [n] | sources | library",
        "  — fiction studio (genre engines, novel|serial) —",
        "  /book fiction <premise> --genre <g> --mode <novel|serial>",
        "  /book fiction-write <slug>              next chapter (validated)",
        "  /book fiction-status <slug>             arcs, threads, cast",
        "  — wisdom keeper (corpus · history · practice) —",
        "  /wisdom ask <query>                     ask the corpus (answer + provenance)",
        "  /wisdom practice list | <session-id>    list sessions / start a guided session here",
        "  /wisdom practice stop                   stop the running session",
        "  /wisdom read <work> [chapter] [page]   read the actual full text, paginated",
        "  /wisdom traditions [tradition]         what each tradition is + its works",
        "  /wisdom seeds                          every seed source + its works",
        "  /wisdom timeline [tradition] [n] | /wisdom compare <topic> | /wisdom status",
        "  — decode & crypto —",
        "  /decode <data> | /decode file:<path>     Universal Decoder (sends binary back)",
        "  /decode hash <digest> | /decode decoders identify hashes / list engine",
        "  /cookies [check]                     YouTube cookie file status / setup help",
        "  /cookielab <header> | /cookielab ingest <h>  CookieLab: classify, fingerprint, decode",
        "  /structure <objective>                   structured brief (subgoals, acceptance…)",
        "  /cipher enc <data> with <pass>           AES-256 sealed blob (integrity-tagged)",
        "  /cipher vault put <name> <secret> with <pass>   named secrets vault",
        "  /cipher vault get <name> with <pass> | list | rm <name>",
        "  /cipher dec <blob> with <pass>           open it (rejects wrong pass / tamper)",
        "  /investigate <hash|jwt|cookie|url|…>   one pass: decode → crack → OSINT → graph",
        "  /monitor add <url|file> [every Ns]       alert on change (with diff)",
        "       [--webhook URL] [--min-gap 60]     POST alerts as JSON; throttle",
        "  /monitor alert <ref> [--webhook … --min-gap N]  change a watch's delivery",
        "  /monitor list | tick | rm <ref>          manage watches",
        "  — she speaks first (owner DMs only, never anyone else) —",
        "  /proactive                            push-send switches + delivery states",
        "  /notify [n]                           recent alerts with delivery states",
        "  /email triage|inbox|search|read|thread|drafts|compose|reply|send|approve|discard|followups  full mailbox (Gmail) + triage + draft queue",
        "  /mission status [id|name]             mission progress, ETA, stall reasons",
        "  /mission pause|resume|cancel|retry  pause, resume, stop, or restart a mission",
        "  /mission watch|unwatch <id>          this chat gets milestone updates",
        "  /mission new <research|build|fix>…   create a mission from a template",
        "  env: NM_PARTNER_PROACTIVE_ENABLED=0 silences all pushes;",
        "       NM_PARTNER_PROACTIVE_BRIEFING=0 / _WATCHERS=0 toggle each kind;",
        "       quiet hours NM_PARTNER_QUIET_START/_END (default 22–8)",
        "  — features & arena —",
        "  /features                               list all feature toggles",
        "  /features <name> on|off                 arena|group_posts|proactive_dm|vision|search",
        "  /arena [status|run [topic]|topics|stream [n]|export [n]]",
        "  /arena approve <id> | /arena deny <id>  review its builds",
        "  /upgrade list                         research findings awaiting your review",
        "  /upgrade show <id> | /upgrade diff <id>  full ticket · preview the actual patch",
        "  /upgrade approve <id>                 test-gated apply, then a what-changed digest",
        "  /upgrade deny <id> <reason> | /upgrade applied",
        "  — trial accounts (one account, delivered to you) —",
        "  /trial start <platform>                 plan one real trial signup",
        "  /trial save <platform> <login> <pass>   store it encrypted",
        "  /trial send <platform>                  send it via WhatsApp/Telegram",
        "  /trial list | /trial rm <platform>",
        "  — temp mail & sms (per-user inboxes: yours alone, never another user's) —",
        "  /tempmail new | status | forget         get a temp inbox (mail.tm → 1secmail → guerrilla)",
        "  /checkmail [read <n> | code | forget]    read your temp inbox",
        "  /vnumber new [country] | status | forget get a temp number for SMS codes",
        "  /checksms [code | forget]               read SMS on your temp number",
        "  — reachability wave (capabilities that had no command) —",
        "  /browser open|text|links|forms|headings <url>   the browser tool, in chat",
        "  /browser takeover [url]   takeover sessions — status / open (owner-only)",
        "  /browser run <url> [goal]  drive a live tab with the agent loop — auto-escalates walls (owner-only, background job)",
        "  /browser run status [job]  job progress — steps, wall outcome, final URL",
        "  /paper start <symbol> [market] [capital] | status [session|symbol] | stop | list",
        "   paper trading — simulated, never real money",
        "  /people list [n] | show <name> | add <name> | note <name> <text>",
        "  /commitments list | add <what> [by <deadline>] | kept|broken|cancel <id> [evidence]",
        "   /commitments overdue | due | reliability [promisee] | verify",
        "  /focus start [label] | pause | resume | stop [note] | status | today | week",
        "  /backup status | verify                backup health (read-only; runs need the vault, stay scheduled)",
        "  — expansion —",
        "  /game [list|<name>|quit]                41 games (DM + group): /game list",
        "  /hangman /mafia /rpg /trivia /spy /wordchain /duel\n"
        "   /king /story /case /world /escape /political /auction\n"
        "   /shop /wyrr /two_truths /numberguess   start any of the 41 (every chat)",
        "  /predict <stake> <over|under> <1-99>    prediction pit — call the d100 roll, win 2x (unlock by beating challenges)",
        "  /game leaderboard|stats|shop|balance    the shared table: rankings, record, coins",
        "  /inventory | /equip <gear> | /unequip [slot] | /repair <gear>",
        "   arena gear — persistent swords & armor with durability, grades, sets",
        "  /level                                   XP, level, arena stat growth",
        "  /npc list | /npc talk <name> <msg> | /npc mood <name>",
        "   the living cast — NPCs with memory, mood, and boundaries",
        "  /dm mood [grim|whimsical|epic|deadpan|neutral]",
        "   the narrator's persona for this game",
        "  /news [run|status]                      fetch + summarize the feeds",
        "  /research [run [domain]|status]         lifestyle | tech | cyber",
        "  /code <what to build>                   the coding bot (draft→run→fix)",
        "  /checkpoint [label]                   snapshot the coding workdir",
        "  /rewind [n]                           restore a checkpoint (n=1 latest)",
        "  /checkpoints                          list saved checkpoints",
        "  /py <python code>                       run it sandboxed; -s name keeps a session, -r resets",
        "  /remember <text> [kind] [tags:a,b]      store it in her long-term memory",
        "  /recall [query]                         what she remembers (top 5)",
        "  /forget <id or description>             delete a memory",
        "  /memories [name]                        trust view: people tracked, memory counts",
        "  — voice / screen / tools —",
        "  /tts [speed=1.0] [pace=natural] [gap=0] <text>  speak it aloud (audio sent back) · /tts engines",
        "  /stt <path>                             transcribe an audio file",
        "  /look <path|url> [focus]                screen-reader analysis of a screenshot",
        "  /schedule add <name> <when> <action>    at 2026-01-01 09:00 | every 30m | 22:00",
        "  /schedule list | rm <name> | enable <name> | disable <name> | run <name>",
        "  /db tables | schema <table> | query <sql> | counts",
        "  /api list | <connector> [json]          external APIs (weather, fx, github, …)",
        "  /connectors [list] | status <name>     manage the service connectors",
        "  /connectors connect <name>             guided connect (key via env var, never chat)",
        "  /notion dbs|query|add · /gcal [agenda]|add · /trello boards|lists|cards|add",
        "  /exness balance|positions|price|buy|sell|close · /stripe balance|customers|charges|link",
        "  /swarm <goal> [workers]                 parallel devon agents + fusion",
        "  — network / proxy / osint / automation —",
        "  /dns <domain> [record]                  A/AAAA/MX/NS/TXT/SPF/CAA (no deps)",
        "  /scan <target> [ports] [banner]         port scan — own infra only",
        "  /whois <domain>                         registration data (RDAP)",
        "  /ports                                  what is listening on this machine",
        "  /proxy status|list|test|set <url>|clear route outbound traffic (your proxies)",
        "  /proxy scrape|refresh|pool [scheme] free-proxy lab  /proxy rotate on [strategy]",
        "  /proxy health — pool by protocol/anonymity/country, latency, refresh, backoffs",
        "  /proxy pick [udp] [country=XX] [anon=elite] — protocol-aware pick",
        "  /proxy affinity bind|lookup|release|list — sticky proxy per site",
        "  /proxy discover [seed urls] learn new list sources from the internet",
        "  /proxy sources — health of every source (retired ones auto-retry)",
        "  /proxy schedule on [min] — auto re-check the pool every N minutes",
        "  /proxy ssh start <name> <host> <user> [key] [port] — SSH → SOCKS5 tunnel",
        "  /workspace [status|scale <n>|up|down|pause|resume|add|remove] the virtual CPU farm",
        "  /gen <kind> <name> [json]               generate a validated script",
        "  /osint <domain|ip|url|email>            read-only public-intel report",
        "  /osint campaign <seeds…>                automated investigation walk",
        "  /osint graph clusters|node|merge|stats  identity correlation graph",
        "  /osint graph decoder <report.json>     feed decoder findings into graph",
        "  /record start <name>|stop|step <t> [json]  capture actions into a macro",
        "  /macro <name> [json]                    replay a recorded macro",
        "  /file <platform> <chat> <path> [caption] send a file to any live chat",
        "  /publish <platform> <chat> <md> [fmt]   make md→pdf/html/… and send it",
        "  /deliver report <topic> [--section \"T::body\"]  styled report → zip → send here",
        "  /data mine [name] | list | fetch <name>  train data (mine/free HF sets)",
        "  /evolve <instruction> | apply <id> | list   self-improvement (tests-gated)",
        "  /evolve audit | research <t> | revert <id> | auto [n] | queue add <goal>",
        "  /evolve git | publish [branch] [--push]   where commits land + make permanent",
        "  /speak <text>                            I say it as a voice note",
        "  /voice list|use|say|clone…                the voice catalogue (switch voices live)",
        "  /bet analyze <home> vs <away> [h d a]     ensemble ML match analysis + value",
        "  /bet bankroll [set <amt>] | backtest [n]  bankroll + walk-forward backtest",
        "  /task add <instruction> | /task run [id] | /task list",
        "  /notify [n]                             recent alerts",
        "  /mission status [id|name]               mission progress, ETA, stall reasons",
        "  /mission pause|resume|cancel|retry    pause, resume, stop, or restart a mission",
        "  /mission watch|unwatch <id>            this chat gets milestone updates",
        "  /mission new <research|build|fix>…     create a mission from a template",
        "  /image <path-or-url> | <prompt>         look up, or generate an image",
        "  /lens <path-or-url>                     reverse image search",
        "  — devon (autonomous dev agent) —",
        "  /devon <free text>                      e.g. 'check if the brain replied to the last messages'",
        "  /devon                                  what he's on / recent digests",
        "  — reasoning (shows its work) —",
        "  /think <question>                       step-by-step answer with the full trace",
        "  /think <question> <strategy>            strategy: cot|decompose|hypothesize|critique|tree|auto",
        "  /benchmark [dimension]                  how sharp the system is right now (0-1)",
        "  /redteam [scenario]                   attack my own loop (sandboxed) and report holes",
        "  — media system (music · playback · video) —",
        "  /music <topic> [style] [vocals|novocals] compose a real song (lyrics + MIDI + TTS hook vocal)",
        "  /music styles | /music song [slug]      browse styles / re-fetch a saved song",
        "  /music bed <topic> [style]              AI instrumental bed (ACE-Step, needs GPU)",
        "  /music full <topic> [style] [voice]     full song: bed + AI vocals (DiffSinger→RVC)",
        "  /music voices [add <id> <pth> <src>]     list / register RVC voices",
        "  /distribute [song.wav]                  release walk-through: splits (mandatory) + AI disclosure + checklist",
        "  /distribute legal                       legal weather: AI disclosure rules, Content ID, distributor terms",
        "  /play <paths…>                          queue + play audio (mpv when installed)",
        "  /play status | queue | pause | resume | stop | next | prev",
        "  /play seek <s> | volume <n> | remove <n> | clear",
        "  /video <query> [platform]               find videos across the web (ranked)",
        "  /video download <url> [audio]           download it (yt-dlp)",
        "  /director cover <script>                professional coverage (verdict: pass/consider/recommend)",
        "  /director plan <script> [--style noir]  breakdown + lighting/camera design per shot",
        "  /director shoot <production-id>         generate the shots (videogen) + frame/Seer QA",
        "  /director regen <prod-id> <shot-id>     one regeneration for a failed shot",
        "  /director show [production-id]          production + QA status",
        "  — anime (AniList records, MyAnimeList cross-check) —",
        "  /anime <title> | /manga <title>           the record: score, episodes, studio, cover",
        "  /anichar <name> | /staff <name> | /studio <name>   people & studios",
        "  /animechars <title> | /episodes <title>   cast · episode list + guide",
        "  /trailer <title> | /streaming <title>     trailer · legal watch links",
        "  /animereview <title> | /recommend <title> scores + reviews · more like it",
        "  /nextep <title>                           next episode: WAT time + countdown",
        "  /anischedule                              the week's airing grid (WAT)",
        "  /airing | /upcoming | /season [s] [y]     airing now · announced · one season",
        "  /topanime | /topmanga | /topchars         all-time best, by the source's own ranks",
        "  /popular [manga] | /trending [manga]      most popular · trending now",
        "  /genres | /genre <name>                   genre list · top of one genre",
        "  /randomanime | /randommanga | /randomchar a random record",
        "  /animequote [title]                       a quote, attributed",
        "  /favorite [add <t>|remove <n>|open <n>]   your own saved list",
        "  /whatanime                                reply to a frame: show + episode + timestamp",
        "  /waifu | /neko | /kitsune | /husbando | /charpic   pictures",
        "  — execution · archives · builders —",
        "  /exec <code> [lang]                     run code sandboxed (py/js/bash/c/…)",
        "  /exec languages                         what's installed here",
        "  /fix <code> [lang] [--rounds 4]         run it; the model rewrites failures until green",
        "  /zip <paths…> --dest x.zip              create an archive (zip|tar.gz|…)",
        "  /zip list|info|extract <archive>        inspect / crack it open (safe)",
        "  /zip digest <archive>                   extract it and curate every doc into memory",
        "  /apps build <name> --stack flask        scaffold a real runnable app",
        "  /apps [list|stacks|info <name>]         what she has built",
        "  /apps serve <name> [--port N]           run it as a live server (health-checked)",
        "  /apps stop <name> | /apps served        kill it / list live servers",
        "  — media hub (one-call orchestrator) —",
        "  /hub song <topic…> [style]              compose → queue → play a real song",
        "  /hub video <query…> [platform]          find → download → queue → play",
        "  /hub podcast <query…>                   find → download → transcribe → chapters",
        "  /podcast <query…>                       the podcast pipeline (shortcut)",
        "  /hub status                             the player right now",
        "  — discovery —",
        "  /list [group]                           every executable command, categorized",
        "  /commands [group] | /menu [group]       aliases of /list",
        "  — stacking: several commands, one message —",
        "  one command per line — every line must be a full /command:",
        '  e.g. "/status" then "/mood" then "/providers" on three lines',
        '  runs in order, one reply per step under a "── 1/3 /status ──" header',
        '  "{prev}" in a later step is replaced by the previous step\'s output',
        '  ("{prev:1}" = step 1\'s); "{{prev}}" stays literal',
        "  a step that fails, is denied, or asks for confirmation STOPS the stack",
        "  — later steps do not run, and the stop is reported, never silent",
        "  max 5 commands per message; games and quit/forget/restart/deploy/",
        "  update/pull can never be stacked — send those on their own",
        '  ";" and "&&" do NOT stack: they belong to /route, /overview, /exec,',
        "  /py, URLs and quoted args — a newline is the only separator",
        '  inside code fences or "quoted strings", a line is argument text,',
        "  never a new command",
        "  /quit",
    ]
)


@dataclass(frozen=True)
class ControlCommand:
    kind: str
    arg: str = ""    #: first argument ("" when there are none)
    tail: str = ""   #: everything after the command word (for multi-word commands)
    #: the @BotName a group command was addressed to, lowercased
    #: ("" when unaddressed). It names the ONE account the command
    #: is for: the runtime's cross-surface claim reads it so every
    #: other surface stands down (owner order 2026-11-05 — one
    #: command, one handling, across all surfaces).
    addressee: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "arg": self.arg, "tail": self.tail}


def parse_control(text: str) -> ControlCommand | None:
    """Parse a control message; None when it's ordinary conversation.

    ``text`` must be a string — ``None`` is treated as no message (returns
    None). Anything else raises :class:`TypeError`, because message text
    arriving as a non-string is a wiring bug, never valid input.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        raise TypeError(f"parse_control expects str, got {type(text).__name__}")
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    rest = stripped[1:].strip()
    if not rest:
        return ControlCommand(kind="help")
    words = rest.split(None, 1)
    kind = words[0].lower()
    addressee = ""
    if "@" in kind:
        # Telegram's group command menu inserts /cmd@BotName —
        # that IS the command, addressed to the bot (owner
        # live-test 2026-11-03: /work@Nomorals_bot fell into
        # conversation while /work executed).  Strip the addressee
        # from the kind — but KEEP it on the command: it names the
        # one surface that owns this command, and the runtime's
        # cross-surface claim stands every other surface down
        # (2026-11-05: the userbot executed /rmdeleted@Nomorals_bot
        # alongside the bot and contradicted it in the room).
        kind, _, addressee = kind.partition("@")
        addressee = addressee.strip().lower()
    if kind not in CONTROL_COMMANDS:
        return None  # unknown slash: treat as a normal message, let her answer
    tail = words[1].strip() if len(words) > 1 else ""
    arg = tail.split()[0] if tail else ""
    n = len(tail.split()) if tail else 0
    min_args, max_args = CONTROL_COMMANDS[kind]
    if n < min_args:
        return ControlCommand(kind="error", arg=f"/{kind} needs {min_args} argument(s) — /help {kind} shows the usage", addressee=addressee)
    if max_args is not None and n > max_args:
        return ControlCommand(kind="error", arg=f"/{kind} takes at most {max_args} argument(s) — /help {kind} shows the usage", addressee=addressee)
    return ControlCommand(kind=kind, arg=arg, tail=tail,
                          addressee=addressee)


# ── command stacking ───────────────────────────────────────────────────────
#: Maximum commands in one stacked message. Stacks are for quick chains
#: (status → mood → providers), not scripts — deeper chains split into
#: smaller stacks. Depth is enforced in ``handle_control``.
MAX_STACK_DEPTH = 5

#: Cap on a previous step's output when ``{prev}`` expands into a later
#: step — without it, one long output feeding the next step could blow
#: the next command's argument up without bound.
PREV_PLACEHOLDER_MAX_CHARS = 2000

#: Commands that can never appear in a stack, whatever the caller.
#: * games (every GAME_COMMANDS kind + the game hub / npc / charagent /
#:   dm / predict): interactive and stateful — one game per message,
#:   and a stack must never auto-advance a game.
#: * confirmable + lifecycle (quit/forget/restart/deploy/update/pull):
#:   confirmation and shutdown/redeploy stay one-message-per-command;
#:   a stack must never arm or answer a confirmation for the owner.
#: Everything else stacks, including long-running commands (book, music,
#: searchdeep, …) — they run in order and delay the later steps, which
#: is the documented cost, not a silent failure.
NON_STACKABLE_COMMANDS: frozenset[str] = frozenset({
    *GAME_COMMANDS,
    "game", "npc", "charagent", "dm", "predict",
    "quit", "forget", "restart", "deploy", "update", "pull",
    "provision",
})

#: The exact owner-grant denial :func:`handle_control` returns. The stack
#: runner compares against it so a denied step stops the stack instead of
#: being mistaken for a normal step reply.
STACK_DENIAL_REPLY = "that one's just for the owner."


def _split_stack_lines(text: str) -> list[str]:
    """Split ``text`` on newlines that are NOT inside a protected span.

    Protected spans: fenced code blocks (``` / ~~~ runs of 3+, opened
    mid-line or at line start), double-quoted strings, single-quoted
    strings (a quote only opens after start/whitespace/``=(:,[{`` so
    ``don't``/``it's`` apostrophes never open one), and inline `code`
    spans. Newlines inside any of them are argument text, never command
    boundaries. Unbalanced spans swallow the rest of the message — the
    safe fallback, since the whole message then parses as one command.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    segments: list[str] = []
    cur: list[str] = []
    fence: str | None = None      # "`" or "~" while inside a fence
    quote: str | None = None      # '"' or "'" while inside quotes
    inline = False                # inside a `...` / ``...`` span
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if fence is not None:
            if ch == fence:
                j = i
                while j < n and text[j] == ch:
                    j += 1
                if j - i >= 3:
                    cur.append(text[i:j])
                    i = j
                    fence = None
                    continue
            cur.append(ch)
            i += 1
            continue
        if quote is not None:
            if ch == "\\" and quote == '"' and i + 1 < n:
                cur.append(text[i:i + 2])
                i += 2
                continue
            if ch == quote:
                if quote == "'":
                    nxt = text[i + 1] if i + 1 < n else ""
                    if nxt and nxt.isalnum():
                        # Apostrophe inside a quoted word, not a closer.
                        cur.append(ch)
                        i += 1
                        continue
                cur.append(ch)
                quote = None
                i += 1
                continue
            cur.append(ch)
            i += 1
            continue
        if inline:
            if ch == "`":
                cur.append(ch)
                inline = False
                i += 1
                continue
            cur.append(ch)
            i += 1
            continue
        # Normal text.
        if ch in ("`", "~"):
            j = i
            while j < n and text[j] == ch:
                j += 1
            run = j - i
            if run >= 3:
                fence = ch
                cur.append(text[i:j])
                i = j
                continue
            if ch == "`":
                inline = True
                cur.append(text[i:j])
                i = j
                continue
            cur.append(text[i:j])
            i = j
            continue
        if ch == '"':
            quote = '"'
            cur.append(ch)
            i += 1
            continue
        if ch == "'":
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < n else ""
            if ((prev == "" or prev.isspace() or prev in "=(:,[{")
                    and nxt and not nxt.isspace()):
                quote = "'"
                cur.append(ch)
                i += 1
                continue
            cur.append(ch)
            i += 1
            continue
        if ch == "\n":
            segments.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    segments.append("".join(cur))
    return [s.strip() for s in segments if s.strip()]


def split_command_stack(text: str) -> list[str] | None:
    """Split a stacked message into its commands, or None when it is not one.

    A message is a stack iff it has 2+ non-blank lines (after removing
    newlines inside fenced blocks / quoted strings / inline code) AND
    every line parses as a known command via :func:`parse_control`.
    Anything else returns None and the caller keeps the existing
    single-command path — so multi-line single commands (``/exec`` code,
    a ``/say`` whose second line is prose) never change behavior.

    ``;`` and ``&&`` are deliberately NOT separators: they are live
    syntax inside /route, /overview (``;;``), /exec, /py, URLs and
    quoted args. The newline syntax is the one split that cannot
    misfire on any of those.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        raise TypeError(
            f"split_command_stack expects str, got {type(text).__name__}")
    parts = _split_stack_lines(text)
    if len(parts) < 2:
        return None
    for part in parts:
        cmd = parse_control(part)
        if cmd is None or cmd.kind == "error":
            return None
    return parts


_PREV_RE = re.compile(r"\{prev(?::(\d+))?\}")
_PREV_ESCAPE = "\x00STACK-PREV\x00"


def expand_prev_placeholders(segment: str, outputs: list[str]) -> str:
    """Expand ``{prev}`` / ``{prev:N}`` in a stack segment.

    ``{prev}`` → the previous step's output; ``{prev:N}`` → step N's
    output (1-based). Out-of-range references and ``{prev}`` in step 1
    stay literal — silently emptying them would corrupt the command.
    ``{{prev}}`` is the escape for a literal ``{prev}``. Expansions are
    truncated to :data:`PREV_PLACEHOLDER_MAX_CHARS` so one long output
    cannot blow up the next step's argument.
    """
    def _repl(m: re.Match) -> str:
        if m.group(1) is not None:
            idx = int(m.group(1)) - 1
            if idx < 0 or idx >= len(outputs):
                return m.group(0)
            src = outputs[idx]
        else:
            if not outputs:
                return m.group(0)
            src = outputs[-1]
        if len(src) > PREV_PLACEHOLDER_MAX_CHARS:
            # Spec §6.1: truncation is always noted WITH a count.
            src = (src[:PREV_PLACEHOLDER_MAX_CHARS]
                   + f"… (truncated, {len(src) - PREV_PLACEHOLDER_MAX_CHARS} more chars)")
        return src

    seg = segment.replace("{{prev}}", _PREV_ESCAPE)
    seg = _PREV_RE.sub(_repl, seg)
    return seg.replace(_PREV_ESCAPE, "{prev}")


def stack_header(index: int, total: int, kind: str) -> str:
    """One step's header in a combined stack reply: ``── 1/3 /status ──``."""
    return f"── {index}/{total} /{kind} ──"


def help_text() -> str:
    """The catalog — plus any command registered after the text was written.

    The games registry grows (poker was a late addition) and the static help
    drifted from CONTROL_COMMANDS; rather than chase every line, coverage is
    closed here — every registered command is guaranteed to appear.
    """
    text = _HELP_TEXT
    missing = sorted(kind for kind in CONTROL_COMMANDS if f"/{kind}" not in text)
    if missing:
        text += "\n  /" + " /".join(missing) + "   (newer games — /list has the full detail)"
    return text


#: Section keyword → emoji for the styled help menu.
_HELP_SECTION_EMOJI: list[tuple[str, str]] = [
    ("control commands", "🎛️"),
    ("search", "🔍"),
    ("bookforge", "📚"),
    ("wisdom", "🧠"),
    ("decode", "🔐"),
    ("crypto", "🔐"),
    ("speaks first", "📣"),
    ("proactive", "📣"),
    ("features", "🏟️"),
    ("arena", "🏟️"),
    ("trial", "🧪"),
    ("expansion", "🎮"),
    ("game", "🎮"),
    ("voice", "🎙️"),
    ("screen", "🎙️"),
    ("tools", "🧰"),
    ("network", "🌐"),
    ("proxy", "🌐"),
    ("osint", "🌐"),
    ("automation", "🌐"),
]


def _help_section_emoji(title: str) -> str:
    low = (title or "").lower()
    for keyword, emoji in _HELP_SECTION_EMOJI:
        if keyword in low:
            return emoji
    return "📌"


def help_menu(platform: str = "telegram") -> str:
    """The help catalog rendered as a styled, scannable menu.

    ONE declared structure (:func:`menu_sections` — the registry
    itself, so command coverage can never drift) rendered through
    the shared typography layer for every platform (2026-11-06
    menu unification): WhatsApp gets its XMD frame
    (:func:`whatsapp_menu`), every other platform the layer's
    ``menu``-class Card render — the same sections, order and
    titles, adapted by the layer to the surface's markup
    discipline (HTML on Telegram, plain where the surface is
    plain), stamp included. Before this, the Telegram menu was
    parsed by hand out of a hand-kept text and rendered by a
    separate style path — two structures, two systems. Never
    raises.
    """
    try:
        if str(platform or "").strip().lower() in ("whatsapp", "wa"):
            # WhatsApp gets its OWN frame (owner rule) over the
            # SAME declared sections — text-first, no
            # inline-button assumptions.
            return whatsapp_menu()
        from .typography import Card, render_card

        card = Card("menu", title="Devon — command menu", icon="✨")
        card.line("you don't need to memorize these — just say "
                  "what you want in plain words.")
        for emoji, title, items in menu_sections(platform):
            if not items:
                continue
            card.block_lines(
                [f"{cmd} — {desc}" if desc else cmd
                 for cmd, desc in items],
                title=f"{emoji} {title}")
        card.note("say it in plain words, or tap a command — "
                  "/help <command> for detail")
        return render_card(card, platform=platform).text
    except Exception as exc:  # noqa: BLE001
        record_failure('control.help_menu', exc)
        return help_text()


#: THE menu's declared section order + short titles: (LIST_GROUPS
#: name, display title). Commands still come from LIST_GROUPS at
#: render time — this table only sets the order and the short
#: titles, and it is canonical for EVERY platform's catalog menu
#: (2026-11-06 menu unification: it began as the WhatsApp menu's
#: curation; the Telegram menu rendered a different structure from
#: a hand-kept text, which is how the two menus drifted into
#: different systems). An entry naming a group that no longer
#: exists is skipped; LIST_GROUPS entries missing here are
#: appended at the end in registry order, so the menu can never
#: drift from the registry.
_MENU_SECTION_ORDER: list[tuple[str, str]] = [
    ("voice & vision", "voice & vision"),
    ("media system — music · playback · video · podcast", "media"),
    ("search & research", "search"),
    ("wisdom keeper — corpus · history · practice", "wisdom"),
    ("games — 41, DM + group, start them directly", "games"),
    ("memory & thinking", "memory"),
    ("building for real — code & missions", "build"),
    ("tools & automation", "tools"),
    ("her day — state & control", "control"),
    ("discovery", "discovery"),
]


def menu_sections(
        platform: str = "",
) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """THE declared catalog-menu structure — one section list for
    every platform (2026-11-06 menu unification).

    Sections come from the registry at call time (LIST_GROUPS,
    one-liners from LIST_ONELINERS / COMMAND_DETAILS), ordered and
    titled by :data:`_MENU_SECTION_ORDER`, so coverage can never
    drift from what the system actually runs. Platform differences
    are declared HERE, once, and are platform-truth only:

    * WhatsApp leads with its plain-words group section (those
      capabilities exist only on the WhatsApp bridge) and omits
      :data:`WHATSAPP_UNSUPPORTED_KINDS` — kinds that surface
      cannot execute at all.
    * Every other platform renders the full registry set.

    A trailing "more" section catches any command ``help_text()``
    knows that no section claimed. Returns
    ``[(emoji, title, [(command, description), ...]), ...]``.
    Renderers (the Telegram Card render, the WhatsApp XMD frame)
    consume this one structure — a menu's sections, order and
    titles are the same on every platform by construction.
    """
    on_wa = str(platform or "").strip().lower() in ("whatsapp", "wa")
    sections: list[tuple[str, str, list[tuple[str, str]]]] = []
    if on_wa:
        from . import whatsapp as _wa  # local: never at module load
        sections.append((
            "💬", "groups — in plain words",
            [(f'"{say}"', what)
             for _, say, what in _wa.GROUP_CAPABILITIES],
        ))
    by_name = dict(LIST_GROUPS)
    ordered = [g for g, _ in _MENU_SECTION_ORDER if g in by_name]
    ordered += [g for g, _ in LIST_GROUPS if g not in ordered]
    titles = dict(_MENU_SECTION_ORDER)
    for group in ordered:
        title = titles.get(group, group)
        emoji = _GROUP_EMOJI.get(group, "📌")
        items: list[tuple[str, str]] = []
        for kind in by_name[group]:
            if kind not in CONTROL_COMMANDS:
                continue
            if on_wa and kind in WHATSAPP_UNSUPPORTED_KINDS:
                # Not on this surface at all: WhatsApp cannot
                # execute it (no per-person mute — see the
                # constant). Absent from the menu, and from the
                # coverage sweep below.
                continue
            one = LIST_ONELINERS.get(
                kind, COMMAND_DETAILS.get(kind, {}).get("what", ""))
            items.append(("/" + kind, one))
        sections.append((emoji, title, items))
    # coverage: every command help_text() knows must appear somewhere
    shown = {cmd for _, _, items in sections for cmd, _ in items}
    missing = sorted(
        kind for kind in CONTROL_COMMANDS
        if not (on_wa and kind in WHATSAPP_UNSUPPORTED_KINDS)
        and f"/{kind}" not in shown and f"/{kind}" in help_text())
    if missing:
        sections.append((
            "📌", "more",
            [("/" + kind,
              LIST_ONELINERS.get(
                  kind,
                  COMMAND_DETAILS.get(kind, {}).get("what", "")))
             for kind in missing],
        ))
    return sections


# ── the live bot-DM menu session (2026-11-06 menufix) ─────────────
#
# The bot-DM menu is ONE live message: the home index, category
# pages and page turns all render from the canonical
# :func:`menu_sections` structure and ride the live-session
# machinery (declared ``session="menu"``), so a tap EDITS the menu
# message instead of posting the catalog as a new raw text wall.
# Callbacks stay inside the list grammar — ``list <section>
# [page]`` — which the control parser and the gated surface's
# callback allowlist already speak.

#: Rendered-page budget for one menu page, with headroom under
#: Telegram's 4096-char message cap for the funnel's stamp/framing.
MENU_SESSION_PAGE_MAX = 3200


def _menu_section_key(title: str) -> str:
    """The one-word callback token naming a menu section (the
    title's first word — ``voice`` for "voice & vision")."""
    words = (title or "").strip().lower().split()
    return words[0] if words else ""


def _menu_session_sections(
        platform: str, public_only: bool,
) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """The canonical sections for a session render, public-filtered
    on the gated surface exactly as :func:`list_catalog` filters."""
    out = []
    for emoji, title, items in menu_sections(platform):
        if public_only:
            items = [it for it in items
                     if it[0].lstrip("/") in PUBLIC_COMMANDS]
        if items:
            out.append((emoji, title, list(items)))
    return out


def _menu_session_index(topic: str, sections) -> int | None:
    """Which section ``topic`` names — the same topic matching
    :func:`list_catalog` applies (aliases, group-name substrings,
    member commands), then section-title / callback-token matches.
    ``None`` when it names nothing (the caller falls back to the
    catalog text path, whose "no group" reply stays the answer)."""
    t = (topic or "").strip().lstrip("/").lower()
    if not t:
        return 0
    want = _LIST_GROUP_ALIASES.get(t, t)
    title_of = dict(_MENU_SECTION_ORDER)
    # Priority matters: a token can be a member command of one group
    # AND the name of another ("memory" is a game kind and a whole
    # section). Exact beats loose — alias-target groups first, then
    # section title / callback token, then list_catalog's loose
    # matching (group-name substring or member command).
    exact = {title_of.get(group, group)
             for group, _kinds in LIST_GROUPS if group.lower() == want}
    for i, (_emoji, title, _items) in enumerate(sections):
        if title in exact:
            return i
    for i, (_emoji, title, _items) in enumerate(sections):
        if t == title.lower() or t == _menu_section_key(title):
            return i
    titles = {
        title_of.get(group, group)
        for group, kinds in LIST_GROUPS
        if t in group.lower().replace(" — ", " ") or group.lower() == want
        or any(t == k for k in kinds)
    }
    for i, (_emoji, title, _items) in enumerate(sections):
        if title in titles:
            return i
    return None


def _menu_session_pages(items) -> list[list[tuple[str, str]]]:
    """Split one section's items into pages under the rendered-size
    budget (lines are ``/cmd — desc`` exactly as the menu renders)."""
    pages: list = []
    current: list = []
    size = 0
    for item in items:
        line_len = len(item[0]) + (len(item[1]) + 3 if item[1] else 0) + 1
        if current and size + line_len > MENU_SESSION_PAGE_MAX:
            pages.append(current)
            current, size = [], 0
        current.append(item)
        size += line_len
    if current or not pages:
        pages.append(current)
    return pages


def _pack_menu_rows(singles: list, per_row: int = 2) -> list:
    """Pack single category buttons into grid rows (owner order
    2026-11-07: the menu is a GRID — categories sit side by side,
    never one stretched full-width line each). Order is preserved
    row-major, so the walk the buttons describe is unchanged."""
    return [list(singles[i:i + per_row])
            for i in range(0, len(singles), per_row)]


def menu_session_render(topic: str = "", page: int = 1, *,
                        platform: str = "",
                        public_only: bool = False,
                        home: bool = False,
                        continuation: bool = False,
                        media: dict | None = None):
    """Render one page of the live bot-DM menu — an ActionableMessage
    declaring ``session="menu"`` — or ``None`` when ``topic`` names
    no section (the caller then answers with the catalog text path).

    ``home=True`` renders the section index (every section + its
    command count). Otherwise the render is one page of one section;
    pages walk the WHOLE catalog linearly (Next on a section's last
    page opens the next section), so the old "All commands" dump is
    a page walk inside this one message — never a second message,
    never a raw slash-list wall. Content comes only from
    :func:`menu_sections`; buttons only from the list grammar.
    ``continuation`` marks tap/topic renders, which edit the live
    menu message in place; the home opener posts it.
    """
    from .tgbot_buttons import actionable
    from .typography import Card, render_card

    sections = _menu_session_sections(platform, public_only)
    if not sections:
        return None
    nav = [[(f"{emoji} {title}", f"list {_menu_section_key(title)}")]
           for emoji, title, _items in sections]
    if home:
        card = Card("menu", title="Devon — command menu", icon="✨")
        card.line("you don't need to memorize these — just say "
                  "what you want in plain words.")
        card.block_lines(
            [f"{emoji} {title} — {len(items)} commands"
             for emoji, title, items in sections],
            title="categories")
        card.note("tap a category — every page lives in this one "
                  "message · /help <command> for the full page on "
                  "one command")
        singles = [row[0] for row in nav]
        if not public_only:
            # Parity with the personal category menu's Power row (an
            # owner-surface affordance; never on the gated bot).
            for i, (_e, title, _i) in enumerate(sections):
                if _menu_section_key(title) == "games":
                    singles.insert(i + 1, ("⚡ Power", "power status"))
                    break
        buttons = _pack_menu_rows(singles)
        buttons.append([("📋 All commands",
                         f"list {_menu_section_key(sections[0][1])}")])
        return actionable(render_card(card, platform=platform).text,
                          buttons, session="menu",
                          session_edit=continuation,
                          media=media)
    idx = _menu_session_index(topic, sections)
    if idx is None:
        return None
    per_section = [_menu_session_pages(items)
                   for _e, _t, items in sections]
    flat = [(si, pi) for si, pages in enumerate(per_section)
            for pi in range(len(pages))]
    page = min(max(1, int(page or 1)), len(per_section[idx]))
    pos = flat.index((idx, page - 1))
    emoji, title, _items = sections[idx]
    card = Card("menu", title=title, icon=emoji)
    card.block_lines(
        [f"{cmd} — {desc}" if desc else cmd
         for cmd, desc in per_section[idx][page - 1]])
    card.note(f"page {pos + 1} of {len(flat)} · "
              "/help <command> for the full page on one command")
    buttons: list = []
    turn: list = []
    if pos > 0:
        psi, ppi = flat[pos - 1]
        turn.append(("◀ Prev",
                     f"list {_menu_section_key(sections[psi][1])} "
                     f"{ppi + 1}"))
    if pos < len(flat) - 1:
        nsi, npi = flat[pos + 1]
        turn.append(("Next ▶",
                     f"list {_menu_section_key(sections[nsi][1])} "
                     f"{npi + 1}"))
    if turn:
        buttons.append(turn)
    buttons.append([("⌂ Menu home", "list home")])
    buttons.extend(_pack_menu_rows([row[0] for row in nav]))
    return actionable(render_card(card, platform=platform).text,
                      buttons, session="menu", session_edit=True)


def _whatsapp_sections() -> list[tuple[str, str, list[tuple[str, str]]]]:
    """The WhatsApp menu's sections — :func:`menu_sections` for
    the WhatsApp surface (kept as the menu builder's own name)."""
    return menu_sections("whatsapp")


def _whatsapp_find_section(sections: list, page: str) -> int | None:
    """Index of the section ``page`` names (number, title, registry name,
    or /list alias). None when it names nothing."""
    key = (page or "").strip().lower()
    if not key or key == "0":
        return None
    if key.isdigit():
        idx = int(key) - 1
        return idx if 0 <= idx < len(sections) else None
    for i, (_, title, _) in enumerate(sections):
        if key in title.lower():
            return i
    registry_name = _LIST_GROUP_ALIASES.get(key)
    if registry_name:
        want = dict(_MENU_SECTION_ORDER).get(registry_name, registry_name)
        for i, (_, title, _) in enumerate(sections):
            if title == want:
                return i
    return None


def _whatsapp_daypart() -> str:
    """The time-aware daypart word for the menu header ("good
    morning" …).  A greeting, never a name: the menu surface does
    not know the requester's display name, so it does not invent
    one.  Never raises."""
    try:
        from datetime import datetime
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "good morning"
        if 12 <= hour < 17:
            return "good afternoon"
        if 17 <= hour < 22:
            return "good evening"
        return "up late"
    except Exception as exc:  # noqa: BLE001
        record_failure('control._whatsapp_daypart', exc)
        return "hello"


def _whatsapp_greeting() -> str:
    """Time-aware menu header. Never raises."""
    try:
        return (f"✨ *Devon — {_whatsapp_daypart()}* · "
                "here's what I can do")
    except Exception as exc:  # noqa: BLE001
        record_failure('control._whatsapp_greeting', exc)
        return "✨ *Devon — command menu*"


def whatsapp_menu(page: str = "") -> str:
    """WhatsApp-native command menu — the XMD frame over the real
    catalog (owner order 2026-11-05: "one Whatsapp bot menu like
    that" — the boxed FEI XMD format, Devon's own content).

    Still data-driven, never a pasted copy of the Telegram menu:
    the first section comes from the WhatsApp adapter's
    GROUP_CAPABILITIES table and every other section comes from
    the command registry (LIST_GROUPS / LIST_ONELINERS /
    COMMAND_DETAILS) via :func:`_whatsapp_sections`, so it can
    never drift from what the system actually runs — and the
    platform-impossible kinds are already absent there.  Only
    the presentation is the pasted format: a header box (name,
    greeting, owner, status, prefix, version, command total,
    mode), one boxed section per category, a powered-by footer.
    The header's total is counted from the rendered lines
    themselves; the version is the repo's own version metadata;
    the footer stamp resolves through the typography theme
    system (surface class ``menu`` — house defaults on this DM
    surface), never a hardcoded string here.  Section TITLES are
    small-capped as presentation; command strings are rendered
    byte-exact, always typable verbatim.

    Honest fields only: the prefix is the parser's real grammar
    (``/``); the greeting carries the daypart, not a name — this
    surface does not know the requester's display name, so it
    does not invent one; the total counts the WhatsApp-visible
    set actually shown, not the global registry.

    * ``page=""`` — the full menu, ONE message: the whole render
      sits far under WhatsApp's 65,536-char protocol cap
      (platform_limits), so no chunking applies on this surface.
    * ``page="3"`` / ``page="voice"`` — just that section, same
      frame.

    Never raises: junk pages fall back to the full menu; total
    failure falls back to :func:`help_text`.
    """
    try:
        from .style import small_caps, xmd_frame_bottom, xmd_frame_line, xmd_frame_top
        from .typography import resolve_style
        sections = _whatsapp_sections()

        def _box(emoji: str, title: str,
                 items: list[tuple[str, str]]) -> list[str]:
            head = f"{emoji} {small_caps(title)}".strip()
            box = [xmd_frame_top(head)]
            for command, desc in items:
                row = f"{command} — {desc}" if desc else command
                box.append(xmd_frame_line(row, "✫✧│"))
            box.append(xmd_frame_bottom())
            return box

        def _stamped(lines: list[str]) -> list[str]:
            style = resolve_style(None, "menu")
            if style.stamp and style.stamp_text:
                lines += ["", f"_{style.stamp_text}_"]
            return lines

        idx = _whatsapp_find_section(sections, page)
        if idx is not None:
            emoji, title, items = sections[idx]
            lines = _box(emoji, title, items)
            lines += ["", "_/help for the full menu · "
                          "/help <command> for one command's "
                          "full page_"]
            return "\n".join(_stamped(lines)).rstrip()

        total = sum(1 for _, _, items in sections
                    for command, _ in items
                    if command.startswith("/"))
        try:
            from ...version import __version__ as _version
        except Exception:  # noqa: BLE001 - version line omits
            _version = ""
        fields = [
            ("👋", "hello",
             f"{_whatsapp_daypart()} — here's what I can do"),
            ("👑", "owner", "peacethefirst"),
            ("⚡", "status", "Online"),
            ("🔰", "prefix", "/"),
        ]
        if _version:
            fields.append(("📦", "version", _version))
        fields += [
            ("📚", "total commands", str(total)),
            ("🔐", "mode", "owner-gated"),
        ]
        lines = [xmd_frame_top(f"🤖 {small_caps('Devon')}")]
        for icon, label, value in fields:
            lines.append(xmd_frame_line(
                f"{icon} {small_caps(label)}: {value}", "✫"))
        lines.append(xmd_frame_bottom())
        lines += ["", "_text-first — no buttons, no tapping needed_"]
        for emoji, title, items in sections:
            lines.append("")
            lines += _box(emoji, title, items)
        lines += ["", "_say it in plain words, or use a command — "
                      "/help <command> gives one command's full page_"]
        return "\n".join(_stamped(lines)).rstrip()
    except Exception as exc:  # noqa: BLE001
        record_failure('control.whatsapp_menu', exc)
        return help_text()


# ── detailed help (wave 67) ─────────────────────────────────────────────────
#: Per-command detail: what it does, exact usage, a real example, and
#: related commands.  ``/help <command>`` in chat (and ``nm help <topic>``)
#: renders one of these; ``/help`` alone renders the catalog.
COMMAND_DETAILS: dict[str, dict[str, str]] = {
    "status": {"what": "her live state — platforms, model budget, active goals, loop cadence.",
               "usage": "/status", "example": "/status",
               "related": "/platforms /benchmark"},
    "ping": {"what": "the life of this very message — when it arrived, when she took it in, when processing started, how long the reply took — plus her system condition.",
             "usage": "/ping", "example": "/ping",
             "related": "/status /platforms"},
    "platforms": {"what": "which chat platforms are running in this session.",
                  "usage": "/platforms", "example": "/platforms",
                  "related": "/start /stop"},
    "start": {"what": "hot-start a chat platform without restarting the system.",
              "usage": "/start <platform>", "example": "/start telegram",
              "related": "/stop /platforms"},
    "stop": {"what": "hot-stop a chat platform (it keeps running data on disk).",
             "usage": "/stop <platform>", "example": "/stop whatsapp",
             "related": "/start /platforms"},
    "mood": {"what": "set her mood — a label (happy, tired, playful, calm, …), or set "
                   "dimensions directly (0-100): affection, happiness, energy, "
                   "trust, jealousy, frustration, intimacy, distance, "
                   "insecurity, pride. 'reset' restores the persona's "
                   "baselines. Bare /mood shows the current state.",
             "usage": "/mood [label]  ·  /mood dim=NN [dim=NN …]  ·  /mood reset",
             "example": "/mood happy   ·   /mood energy=20 frustration=70",
             "related": "/stage"},
    "mode": {"what": "her autonomy level: off (asleep) | suggest (proposes, you approve) | auto (acts).",
             "usage": "/mode off|suggest|auto", "example": "/mode auto",
             "related": "/proposals /power /agent"},
    "agent": {"what": "her agent modes — a combinable set, per chat: adaptive (learns from outcomes and adjusts), autonomous (self-initiates work without being poked), planning (plans first, verifies each step, replans on failure). Bare /agent shows the chat's modes.",
              "usage": "/agent adaptive|autonomous|planning · /agent adaptive+planning · /agent +autonomous · /agent -planning · /agent status · /agent off · /agent tick",
              "example": "/agent adaptive+planning",
              "related": "/mode /mind"},
    "chatmode": {"what": "this chat's reply mode + per-chat feature toggles — the single entry point for group controls. Modes: normal (she answers everything) | games-only (silent for normal chat, game commands still work) | games-off (normal chat works, game commands refused honestly) | quiet (never messages first here, replies when addressed). Features: /chatmode feature <games|music|voice|images|proactive> <on|off> disables one feature in THIS chat only — Group A never affects Group B or DMs. Per-game: /chatmode feature games.<game> <on|off>. Turning games off while a game is live ends it honestly (scores kept). Voice off = text replies only, voice notes still transcribe. Owner-only; run it in the chat to set that chat. In games-only, ALL non-game messages are silenced — including voice notes (even after transcription). Game commands + in-game moves still work. /chatmode normal restores replies. /chatmode list shows every chat with custom settings.",
             "usage": "/chatmode [games-only|games-off|games-on|quiet|normal|status|list] [target]  ·  /chatmode feature <games|music|voice|images|proactive|games.<game>> <on|off> [target]  ·  a target is a chat name, a phone number, or a full key (platform:chat_id); no target = this chat",
             "example": "/chatmode games-off   ·   /chatmode feature music off   ·   /chatmode quiet   ·   /chatmode feature games.chess off   ·   /chatmode games-only Think Creativity   ·   /chatmode list",
             "related": "/status /game"},
    "doctor": {"what": "system health — the Telegram twin of nm doctor: registry (every command dispatched + documented), config (ignored keys must be 0), stores (chat-modes + style-overrides + database), adapters (running state + real error), providers (LLM chain keys present). Each check passes or fails with the real cause. Owner-only.",
             "usage": "/doctor", "example": "/doctor",
             "related": "/status /chatmode"},
    "style": {"what": "inspect + tune the mined style profile for this chat — the numbers Devon replies with (emoji budget, length target, sections/lists/markup), measured from YOUR own messages, never other people's. /style shows the live profile + any overrides + where each override came from; /style <field> <value> sets an override delta (emoji_cap 0..3, target_chars, max_chars, sections/lists/markup on|off); /style reset clears this chat's overrides; /style list shows every chat with overrides. Owner-only.",
             "usage": "/style  ·  /style <emoji_cap|target_chars|max_chars|sections|lists|markup> <value> [target]  ·  /style reset [target]  ·  /style list  ·  a target is a chat name, a phone number, or a full key (platform:chat_id)",
             "example": "/style   ·   /style emoji_cap 0   ·   /style target_chars 200   ·   /style reset",
             "related": "/chatmode /mood"},
    "model": {"what": "switch the active model live (up to 5 fallbacks); no argument shows the chain.",
              "usage": "/model [provider [fallback …]]", "example": "/model local",
              "related": "/status /providers"},
    "providers": {"what": "probe every registered provider and report live/dead with the reason — "
                          "the diagnostic for a silent brain.",
                  "usage": "/providers", "example": "/providers",
                  "related": "/model /status"},
    # /say is ONE command with three forms (single entry — a duplicate key
    # here would silently discard one form, and did: the rich-voice page
    # used to shadow this one while the runtime ran the gateway form).
    "say": {"what": "three forms: send a message to a chat as her, from your hands — speak text as a voice note in a named voice — or rich voice with direction/accent/ambience.",
            "usage": "/say <platform:chat> <text> | /say <voice> <text> | /say [direction] [accent=…] [ambient=…] <text>",
            "example": "/say telegram:123 hi — or — /say trump hello there — or — /say [whispers fearfully] don't go in there",
            "related": "/file /publish /voices /voice /tts /sing"},
    "proposals": {"what": "list everything waiting for your approval: proactive actions + research upgrade proposals.",
                  "usage": "/proposals", "example": "/proposals",
                  "related": "/approve /deny /upgrade"},
    "approve": {"what": "approve one pending proposal so she executes it (upgrade tickets are approved + test-gated applied).",
                "usage": "/approve <id>", "example": "/approve p3",
                "related": "/proposals /deny"},
    "deny": {"what": "deny one pending proposal (upgrade tickets need a reason).",
             "usage": "/deny <id> [reason]", "example": "/deny upg_abc not now",
             "related": "/proposals /approve"},
    "stage": {"what": "set the relationship stage (committed|dating|…), which tunes her behavior.",
              "usage": "/stage [stage]", "example": "/stage dating",
              "related": "/mood"},
    "power": {"what": "power mode: no human-like pacing, maximum capability — needs the key you set in config, or the owner seal (identity + passphrase).",
              "usage": "/power on <key> | /power on <identity> <passphrase> | off | status", "example": "/power on mykey",
              "related": "/mode /features"},
    "quit": {"what": "shut the whole system down (everything is already on disk).",
             "usage": "/quit", "example": "/quit"},
    "search": {"what": "quick research: searches and replies with a cited summary.",
               "usage": "/search <query>", "example": "/search llama 3.1 8b quant",
               "related": "/searchdeep /searchhist"},
    "searchdeep": {"what": "deep multi-query research with synthesis (power mode).",
                   "usage": "/searchdeep <query>", "example": "/searchdeep eBPF rootkit detection",
                   "related": "/search /searchhist"},
    "searchleads": {"what": "report on legit paid-task platforms — what is real, what pays.",
                    "usage": "/searchleads", "example": "/searchleads",
                    "related": "/search /money"},
    "money": {"what": "hunts every kind of money-making opportunity — paid tasks, referrals, free courses, bounties, gigs — ranked by value vs effort.",
              "usage": "/money [scan [kind] | list | new | profile]",
              "example": "/money scan bounty",
              "related": "/searchleads"},
    "searchhist": {"what": "your recent research runs.",
                   "usage": "/searchhist [n]", "example": "/searchhist 5",
                   "related": "/search"},
    "redteam": {"what": "defensive self-attack: runs attack scenarios against a sandboxed copy of my own agent loop and reports the holes it finds. Never touches external systems.",
                "usage": "/redteam [scenario_id]", "example": "/redteam",
                "related": "/benchmark"},
    "predict": {"what": "the prediction pit: call a d100 roll over/under a line, stake coins, correct calls pay 2x. Unlock: earn 'High Roller' (craps) or 'Boss Hunter' (raid boss) — progression-gated, the owner always has access.",
                "usage": "/predict <stake> <over|under> <1-99>", "example": "/predict 50 over 60",
                "related": "/game /bet /achievements"},
    "miniapp": {"what": "group mini-apps: polls, quizzes, shared-expense tracking with settle-up, and event RSVPs — plus wave 2 (AA-04): carpools, rotas, a marketplace, leagues, and split bills. Group chats only — state is group-scoped and isolated from my private memory.",
                "usage": "/miniapp new <poll|expenses|rsvp|quiz> <title> [options…] | /miniapp list | /miniapp show <id> | /miniapp vote <id> <n> | /miniapp expense <id> <amount> <what> | /miniapp rsvp <id> yes|no|maybe | /miniapp carpool|rota|market|league|split … (type the family name alone for its help)",
                "example": "/miniapp new poll \"Best day?\" Mon Tue Wed",
                "related": "/game"},
    "panel": {"what": "persistent live-data panels: user-described mini-apps wired to connectors (fitness, finance, …). Panels refresh on a cadence and answer questions in chat — \"how's my marathon training going?\".",
                "usage": "/panel new \"<name>\" <connector> [hourly|daily|weekly|manual] | /panel list | /panel show <id> | /panel refresh <id> | /panel ask <id> <question> | /panel rm <id>",
                "example": "/panel new \"Marathon training\" strava daily",
                "related": "/miniapp"},
    "gtrip": {"what": "group-travel stack: polls with deadlines, shared expenses with minimized settle-up, per-person budgets, per-traveler flight legs, proposals vs agreed items, merged itinerary. Group chats only — state is group-scoped and isolated from my private memory.",
                "usage": "/gtrip new <name> | /gtrip poll <q> | <opt1> | <opt2> [deadline 2h] | /gtrip vote <id> <opt> | /gtrip expense <amount> <what> [for <m1,m2>] | /gtrip settle | /gtrip leg <flight> | /gtrip propose <idea> | /gtrip agree <id> | /gtrip itinerary",
                "example": "/gtrip poll where should we eat? | suya spot | pizza place",
                "related": "/miniapp /trip"},
    "contract": {"what": "contract risk review — paste a tenancy, employment, or freelance agreement and get a letter grade (A–F) plus flagged clauses explained in plain language. Legal information, never legal advice; Devon is not a lawyer. Owner only.",
                "usage": "/contract review [tenancy|employment|freelance] <paste contract text> | /contract types",
                "example": "/contract review tenancy This tenancy agreement between landlord and tenant...",
                "related": "/legal"},
    "legal": {"what": "legal information hub — contract review (Nigerian playbooks), portfolio obligations, regulatory monitoring, plain-language aid in English, Pidgin, Yoruba, Hausa, or Igbo, grounded research, and precedent citations. Legal information, never legal advice; Devon is not a lawyer. Owner only.",
                "usage": "/legal contract review [tenancy|employment|freelance] <paste contract text> | /legal portfolio | /legal regulatory regulators | /legal aid [en|pcm|yo|ha|ig] <question> | /legal research <question> | /legal citations | /legal [en|pcm|yo|ha|ig] <question>",
                "example": "/legal contract review tenancy <paste agreement>",
                "related": "/contract /contracts /regwatch /learn"},
    "immune": {"what": "defensive immune system — thymus negative selection, innate scan, antibody memory, seven-rung revoke-only police, fever, drift, case files. Owner only. Never acquires capability; owner work is report-only.",
                "usage": "/immune status | scan <text> | antibodies list | add <pattern> <reason> | retire <id> <reason> | test <pattern> | quarantine <artifact> | disarm <grant-id> | arrest <process-id> [system] | fever status | start <reason> <seconds> | end | drift add <tokens> <spawned> <grants> <errors> | report | casefile list | add <text> | seal <id-prefix>",
                "example": "/immune status",
                "related": "/heal /review"},
    "impossible": {"what": "Impossible Made Possible — always-on cross-domain conceive-and-solve: formulate, classify, prior art, TRIZ transfer, feasibility gate, falsifier, attempt. Hardware gets a specification; software gets a build brief. Owner only.",
                "usage": "/impossible solve <problem> [--software] [--fetch-art] | status <id> | attempt <id> | dossier <id> | queue | charter | lattice | examples",
                "example": "/impossible solve a house that stays cool without grid power",
                "related": "/research /code"},
    "learn": {"what": "one front door over learn/ — Socratic/direct tutoring, quiz formats, FSRS flashcards from the mistake notebook, BKT mastery, and WAEC/JAMB syllabus. Owner only.",
                "usage": "/learn tutor [guide me] <topic> | answer <text> | hint | status | stop | /learn quiz formats | /learn flashcards due | stats | list | review <id> <1-4> | export <path> | /learn mastery [weakest] | /learn waec <subject> | waec grade <pct> | /learn jamb <subject> | jamb formats",
                "example": "/learn tutor guide me fractions",
                "related": "/tutor /course /legal"},
    # /research is ONE command (single entry — a duplicate key here used
    # to shadow this page with a stale scheduled-runs page while the
    # runtime ran the web-research handler).
    "research": {"what": "web research — one-shot swarm research on any topic with a brief back in chat, plus the always-on research loop (status/ideas/approve/deny/history/from/ask/docs). For legal questions use /legal instead. Owner only.",
                "usage": "/research <topic> | /research run <topic> | /research status | /research ideas [n] | /research approve <id> | /research history [n]",
                "example": "/research agent coding",
                "related": "/legal /news"},
    "contracts": {"what": "contract portfolio — all your contracts as one obligation tracker: renewals, notice deadlines, payments, SLA terms, all on one timeline with proactive alerts. Legal information, never legal advice; Devon is not a lawyer. Owner only.",
                "usage": "/contracts | /contracts attention [days] | /contracts add <name> | <paste contract text> | /contracts obligations <id> | /contracts track <id> | <desc> | <YYYY-MM-DD> | /contracts done <obligation id> | /contracts sla <id> <metric> | <target> | /contracts breach <id> <metric> | [note] | /contracts timeline",
                "example": "/contracts attention 30",
                "related": "/contract /legal"},
    "regwatch": {"what": "regulatory change monitoring — watch CBN, SEC, NDPA (data protection), FIRS for rule changes affecting your business. Plain-language alerts with official source links. Legal information, never legal advice; Devon is not a lawyer. Owner only.",
                "usage": "/regwatch add CBN [topics…] | /regwatch list | /regwatch remove <id> | /regwatch check | /regwatch regulators",
                "example": "/regwatch add CBN fintech AML",
                "related": "/contracts /research"},
    "uprofile": {"what": "prompt-scaffolded profiles + element-level likes — Hinge-style profiles for gig freelancers, community members, and business listings. Curated prompts fix blank bios; likes attach to specific projects/photos with optional comment openers.",
                "usage": "/uprofile create [gig|community|business] [name] | show <id> | prompt <id> <prompt_id> <answer> | add <id> <type> <title> | like <profile> <element> [comment] | matches | voice <id> [audio_path] [language]",
                "example": "/uprofile create gig Ada — freelance designer",
                "related": "/miniapp /vnote"},
    "vnote": {"what": "voice notes in chat — audio kept for tone, transcripts indexed for search. Full voice language scope: English, Pidgin, Ekiti Yoruba, Hausa, Igbo.",
                "usage": "/vnote send <chat_id> <audio_path> [language] | search <query> [chat_id] | play <note_id> | preview <public|private> <language> <text>",
                "example": "/vnote search price",
                "related": "/uprofile"},
    "match": {"what": "curated daily batches + two-sided stable matching — scarcity-as-ritual picks (daily gig/community/learning highlights) and Gale-Shapley mutual-preference matching with importance-weighted questionnaires and front-loaded deal-breakers.",
                "usage": "/match daily [gig|community|learning] | add <id> | <title> | tags | attrs | questionnaire [surface] | answer <qid> <1-5> | dealbreaker <name> <attr> <pred> <value> | rank <proposer|reviewer> <who> <ids> | run",
                "example": "/match daily gig",
                "related": "/uprofile /miniapp"},
    "cgroup": {"what": "themed community groups + events + IRL meetups — group rooms with topics and post feeds, event lifecycle (create → RSVP → reminders → summary), venue polls, day-of reminders, check-in, weekly rituals, and warm-start discovery. Not owner-gated (shared surfaces).",
                "usage": "/cgroup new <name> | <topic> · join <group_id> · list · post <group_id> <text> · show <group_id> · discover [member] · nudge · schedule <minutes> <text> · posts · devices (device orchestration, AA-18) | /cgroup event new <group_id> | <title> | <YYYY-MM-DD HH:MM> | [where] | [weekly] · rsvp <event_id> <yes|no|maybe> · list · summary · ritual · nudge | /cgroup meetup poll <event_id> | <venue>… · vote · results · attendees · checkin · turnout",
                "example": "/cgroup event new grp_x | Lagos Devs dinner | 2026-10-10 19:00 | Ikeja | weekly",
                "related": "/miniapp /gtrip /match"},
    "challenge": {"what": "photo-proof fitness challenges + streaks + squads — Seer-verified workout photos, leaderboards, daily streaks with loss-aversion nudges, small workout squads (3–8), and challenge-completion badges (features-as-loot). Not owner-gated (shared surfaces).",
                "usage": "/challenge create <name> | <type> | <days> [target N] · join <id> · log <id> · proof <id> <photo-path> · board <id> · streak · risk · squad create <name> | <m1,m2,…> · squad board <squad_id>",
                "example": "/challenge create October Grind | workout-count | 31 target 20",
                "related": "/train /form /cgroup"},
    "room": {"what": "live audio rooms for communities — host → speakers → hand-raise queue → listeners (Clubhouse/Spaces grammar). Rooms are scoped to a community; opt-in recording with always-visible state, live captions on by default, host mic controls + mute/block/report moderation, and tips. Recorded rooms land in the community audio archive (AA-30): recordings, clips, caller-supplied transcripts, speaker stats. Not owner-gated (shared surfaces).",
             "usage": "/room create <community_id> | <title> · list [community_id] · join <room_id> · leave <room_id> · raise <room_id> · lower <room_id> · speak <room_id> <user_id> · drop <room_id> <user_id> · mute|unmute <room_id> <user_id> · block <room_id> <user_id> · report <room_id> <user_id> <reason> · record <room_id> on|off · captions <room_id> on|off · tip <room_id> <user_id> <naira> · show <room_id> · end <room_id> · recordings <community_id> · recording <community_id> <rec_id> · clip <community_id> <rec_id> <start_s> <end_s> <title> · clips <community_id> [rec_id] · attach <community_id> <rec_id> <speaker> | <text> · speakers <community_id> <rec_id>",
             "example": "/room create grp_devs | Friday AMA",
             "related": "/cgroup /match /challenge"},
    "party": {"what": "group parties (AA-05): a DJ request queue with voting, trivia rounds drawn from the group's own quiz mini-apps, a moments wall, RSVPs through the linked event, and an end-of-night recap. Trivia wins pay into the one group economy; a group's first three parties are rehearsals (economy dry-run). The quick variant (/party quick …) spins up a fast playlist party. Group chats only. Not owner-gated (shared surfaces).",
              "usage": "/party new <title> [| <theme>] [in 2h] · list · show · rsvp yes|no|maybe · start · pause · resume · request <track> · vote <track id> · next · trivia · answer <n> · wall <text> · react <post id> <emoji> · end · lastnight · quick new|start|request|vote <n>|next|trivia|answer <n>|wall|end",
              "example": "/party new Friday Turnup | amapiano in 1h",
              "related": "/cgroup /room /miniapp"},
    "ajo": {"what": "ajo / esusu rotating savings circles (AA-22): members contribute a fixed amount each round and one member takes the whole pot in turn order; a full cycle nets everyone to exactly zero. Group chats only — the ledger is the circle's own, under the community dir. Not owner-gated (shared surfaces).",
            "usage": "/ajo new <amount> <circle name> · list · join <id> · show <id> · pay <id> [amount] · payout <id> · remind <id> · balance <id> · order <id> <name1,name2,…> · close <id>",
            "example": "/ajo new 5000 Rent circle",
            "related": "/cgroup /treasury /balance"},
    "book": {"what": "BookForge: writes a real book on a topic (research → outline → "
                     "chapters → PDF with table of contents) and sends the finished "
                     "PDF to you when it's done. Resumable if the run is interrupted.",
             "usage": "/book <topic> [chapters]   |   /book status [slug]  |  /book list",
             "example": "/book eBPF for system security 8",
             "related": "/searchdeep (it researches the topic first)"},
    "books": {"what": "Story reader + continuer (alias of /book for reading): "
                      "follow stories on legal archives (Gutenberg, Royal Road, "
                      "Scribble Hub, AO3, Wattpad, Webnovel), read chapters with "
                      "progress tracking, continue past the ending in the story's "
                      "voice, list sources.",
             "usage": "/books read <url|slug> [n] | /books continue <slug> [n] | /books sources | /books library",
             "example": "/books read https://www.gutenberg.org/ebooks/84",
             "related": "/book /novel"},
    "novel": {"what": "Story reader + continuer: follow webnovels (freewebnovel, pandanovel, "
                      "more), read chapters with progress tracking, and continue any story "
                      "past its ending in the story's own voice (story bible auto-built).",
             "usage": "/novel follow <url|title> | /novel list | /novel read <slug> [chapter] | /novel continue <slug> [n]",
             "example": "/novel continue my-vampire-system 3",
             "related": "/book"},
    "wisdom": {"what": "WisdomKeeper: ask the esoteric corpus (answer + provenance), "
                       "browse the history timeline, or run a guided breathing / "
                       "sitting session right here in chat with timed messages.",
             "usage": "/wisdom ask <query>   |   /wisdom read <work> [chapter] [page]   |   "
                      "/wisdom traditions [tradition]   |   /wisdom seeds   |   "
                      "/wisdom practice list   |   /wisdom practice <session-id>   |   "
                      "/wisdom practice stop   |   "
                      "/wisdom timeline [tradition] [n]   |   /wisdom compare <topic>   |   "
                      "/wisdom status",
             "example": "/wisdom practice box-breathing",
             "related": "/search (general research)"},
    "wis": {"what": "alias of /wisdom.",
            "usage": "/wis <verb…> — same as /wisdom",
            "example": "/wis ask kundalini breathing",
            "related": "/wisdom"},
    "decode": {"what": "Universal Decoder: identifies and decodes almost anything — "
                       "base64/hex/base32/base58/binary, ROT13, leetspeak, URL, "
                       "hashes (with known-secret match), cookies, JWTs, tokens, and "
                       "raw file forensics. Binary results are sent to you as files.",
              "usage": "/decode <data>   |   /decode file:<path>   |   /decode hash <digest>   |   /decode decoders",
              "example": "/decode cGFzc3dvcmQ=",
              "related": "/monitor, nm decode"},
    "cookielab": {"what": "Cookie analysis & handling (the CookieLab): parses a "
                        "Cookie / Set-Cookie header, classifies each cookie "
                        "(session/auth/csrf/tracking/jwt/encoded), fingerprints "
                        "the platforms it implies, decodes opaque values "
                        "(URL, base64, JSON, JWT, hex, gzip), and flags "
                        "cookies missing HttpOnly/Secure. 'ingest' also feeds "
                        "the entities into the knowledge graph.",
             "usage": "/cookielab <cookie-header>   |   /cookielab file:<path>   |   /cookielab ingest <header>",
             "example": "/cookielab PHPSESSID=abc; path=/; HttpOnly; Secure\nJSESSIONID=xyz",
             "related": "/decode, nm cookies"},
    "structure": {"what": "turns a plain-English objective into a structured "
                          "mission brief — intent, ordered subgoals, inputs, "
                          "constraints, acceptance criteria, matching tools, "
                          "and side effects. This is the same brief the "
                          "mission runner plans from, so run it when an "
                          "objective feels too big to just say '/devon do it'.",
             "usage": "/structure <objective>",
             "example": "/structure Build the decoder and verify it against the test corpus",
             "related": "/mission (runs the brief) · /devon"},
    "investigate": {"what": "one-pass investigation pipeline: decode → crack → "
                            "OSINT → knowledge graph. Hand it any artifact — a "
                            "hash, JWT, cookie, URL, encoded blob, or a file — "
                            "and get back what it is, what it decodes to, "
                            "whether it cracks, and who/what it links to.",
                    "usage": "/investigate <hash|jwt|cookie|url|blob|file> [file]",
                    "example": "/investigate 5d41402abc4b2a76b9719d911017c592",
                    "related": "/decode /crack /osint"},
    "monitor": {"what": "Watches a URL or file and alerts you (with a short diff) "
                        "when its content actually changes. Auto-ticked while the bot runs.",
              "usage": "/monitor add <target> [every 300s] [--webhook URL] [--min-gap 60]   |   /monitor alert <ref> [--webhook …] [--min-gap N]   |   /monitor list   |   /monitor tick   |   /monitor rm <ref>",
              "example": "/monitor add https://example.com/status every 300s --webhook https://hooks.example.com/nm --min-gap 300",
              "related": "/schedule, nm monitor"},
    "cipher": {"what": "Real cryptography: AES-256 (PBKDF2) sealed blobs with an "
                       "integrity tag — wrong passphrases and tampered blobs are "
                       "rejected, not returned as garbage.",
              "usage": "/cipher enc <data> with <passphrase>   |   /cipher dec <blob> with <passphrase>   |   /cipher vault put <name> <secret> with <pass>   |   /cipher vault get <name> with <pass>   |   /cipher vault list   |   /cipher vault rm <name> with <pass>",
              "example": "/cipher vault put my-db-secret hunter2 with vaultkey",
              "related": "nm cipher, /decode"},
    "features": {"what": "list or toggle system features: arena, group_posts, proactive_dm, vision, search.",
                 "usage": "/features [name on|off]", "example": "/features arena on",
                 "related": "/arena /power"},
    "arena": {"what": "the content arena: research a topic, stream drafts, review, export, approve builds.",
              "usage": "/arena [status|run [topic]|topics|stream [n]|export [n]|approve <id>|deny <id>|promote <build-id>|ship|apply <proposal-id>|reject <proposal-id> <reason>|scores|sample]",
              "example": "/arena run llama fine-tuning",
              "related": "/features /research"},
    "trial": {"what": "plan / store / send ONE trial-account signup you asked for (stored encrypted, one account); full auto signups (forms, CAPTCHA solver, temp email/SMS) with a confirmation gate.",
              "usage": "/trial [list|start <p>|assist <p> [--yes]|signup <p>|confirm <token>|status|sms [country]|sms code|inbox <service>|save <p> <login> <pass>|send <p>|rm <p>]",
              "example": "/trial list", "related": "/say"},
    "identity": {"what": "the profile bank for signups: your name/email/phone stored once, used by account creation flows.",
              "usage": "/identity [show|set <name|email|phone> <value>|clear]",
              "example": "/identity set name Death", "related": "/trial"},
    "game": {"what": "the social game engine — 41 games across DM, group and "
                     "channel, with a shared economy, items and leaderboards. "
                     "Works for every participant in every chat; in a group a "
                     "new player is seated the moment they speak.",
             "usage": "/game [list|<name>|quit|join|leaderboard [game]|stats "
                      "[name]|balance|shop [buy <item>]]",
             "example": "/game trivia  ·  /game mafia (group)  ·  "
                        "/game leaderboard",
             "related": ""},
    "math": {"what": "Math Class — procedurally generated maths by tier "
                     "(easy → impossible), adaptive tiers, duels, streaks, "
                     "teaching and a leaderboard. A game: works in groups "
                     "for everyone, in DMs for the owner.",
             "usage": "/math · /math easy|medium|hard|expert|master|impossible · "
                      "/math answer <answer> · /math leaderboard · /math streak · "
                      "/math daily · /math duel <opponent> · /math duel "
                      "(matchmaking queue) · /math duel open · "
                      "/math join <code> · "
                      "/math duel_answer <answer> · /math set · "
                      "/math teach <explanation> · "
                      "/math upvote · /math balance",
             "example": "/math  ·  /math answer 42  ·  /math duel @ada",
             "related": "/game · /duel"},
    "npc": {"what": "the living cast — NPCs with personality, memory, mood "
                    "and knowledge boundaries, per game. They remember your "
                    "conversations and react to what happens.",
            "usage": "/npc list · /npc talk <name> <message> · "
                     "/npc mood <name>",
            "example": "/npc list  ·  /npc talk Marlowe \"any advice?\"",
            "related": "/dm · /game"},
    "charagent": {"what": "persistent character agents — they converse, "
                         "remember, and play games with Devon as fellow "
                         "agents, not scripted NPCs. Foundation for podcast "
                         "hosts and room participants.",
            "usage": "/charagent list · /charagent create <name> [trait=0.8] · "
                     "/charagent talk <name> <message> · "
                     "/charagent play <game> <names...> · /charagent forget <name>",
            "example": "/charagent create Zara witty=0.9  ·  "
                       "/charagent play auction Zara Kilo",
            "related": "/npc · /game"},
    "dm": {"what": "the game master's persona — the narrator's mood for "
                   "the current game. Grim, whimsical, epic, deadpan or "
                   "neutral; it shapes how scenes are described.",
           "usage": "/dm mood [grim|whimsical|epic|deadpan|neutral]",
           "example": "/dm mood grim",
           "related": "/npc · /game"},
    "inventory": {"what": "your persistent equipment & consumables — arena "
                          "gear bought from /game shop. Swords and armor "
                          "never vanish: they wear with use, break at 0 "
                          "durability, and can be repaired.",
                  "usage": "/inventory",
                  "example": "/inventory",
                  "related": "equip, unequip, repair, shop"},
    "equip": {"what": "wear a piece of gear into battle — its attack/defense "
                      "applies in the arena immediately.",
              "usage": "/equip <gear>",
              "example": "/equip katana_epic",
              "related": "inventory, unequip, repair"},
    "unequip": {"what": "take gear off (one slot, or everything).",
                "usage": "/unequip [slot]",
                "example": "/unequip weapon  ·  /unequip",
                "related": "equip, inventory"},
    "repair": {"what": "restore a broken or worn piece of gear to full "
                       "durability, for coins.",
                "usage": "/repair <gear>",
                "example": "/repair katana_epic",
                "related": "inventory, equip"},
    "level": {"what": "your persistent progression — XP, level, and the "
                      "stat growth each level brings to the arena. Every "
                      "finished game pays XP; arena wins pay the most.",
              "usage": "/level",
              "example": "/level",
              "related": "inventory, game"},
    "skill": {"what": "learnable battle skills — martial arts for the "
                      "arena. Active skills are cast mid-fight with "
                      "skill <name>; passives are always on. Learning "
                      "costs coins and may need a level.",
              "usage": "/skill [learn <name>] [combos]",
              "example": "/skill  ·  /skill learn dragon_punch  ·  /skill combos",
              "related": "level, inventory, game"},
    "title": {"what": "earnable titles — flair worn next to your name. "
                      "Unlocked through arena achievements and milestones. "
                      "Some titles grant battle buffs.",
              "usage": "/title [set <name>]",
              "example": "/title  ·  /title set dragonslayer",
              "related": "skill, level, game"},
    "stats": {"what": "RPG attributes — strength (attack), stamina "
                      "(HP/defense), mana (skill fuel), intelligence "
                      "(skill power + combo luck). Each level grants "
                      "points to spend.",
              "usage": "/stats [<attribute> [points]]",
              "example": "/stats  ·  /stats strength 2",
              "related": "skill, level, title"},
    "daily": {"what": "the daily hunt — win any arena battle today for "
                      "double XP. One hunt per day.",
              "usage": "/daily",
              "example": "/daily",
              "related": "game, skill, title"},
    "mastery": {"what": "per-game mastery tiers — every non-arena game "
                        "tracks a rank (Novice → Legend, with game-specific "
                        "names) from your wins, games played, and best "
                        "score. Tiers unlock harder sudoku boards, bigger "
                        "gomoku boards, the 2048 marathon, long hangman "
                        "words, trivia sudden death.",
                "usage": "/mastery [game]",
                "example": "/mastery  ·  /mastery sudoku",
                "related": "game, daily, title"},
    "gift": {"what": "send coins, gear, or shop items to another player. "
                     "Two-step with confirmation so nothing moves by "
                     "accident; gifts expire after 5 minutes.",
             "usage": "/gift @name <coins|gear|item> · /gift confirm|cancel|history",
             "example": "/gift @ada 100  ·  /gift @ada katana_rare",
             "related": "game, inventory, level"},
    "mind": {"what": "the Core Mind — the always-on layer that turns a "
                    "natural-language goal into routed work: research swarm, "
                    "builder, browser, downloader, missions, games, "
                    "orchestrator. /mind <goal> forces the routing and shows "
                    "its reasoning; /mind status shows live jobs, pending "
                    "clarifications and the last objective.",
            "usage": "/mind [status|clear|<goal>]",
            "example": "/mind research the history of jazz  ·  /mind status",
            "related": "/profile · /task · /swarm · /game"},
    "wordchain": {"what": "word chain — each word must start with the last letter of the previous one; the house breaks dead ends. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/wordchain", "example": "/wordchain",
             "related": "/game list · /game leaderboard wordchain · /game quit"},
    "hangman": {"what": "hangman — the house picks a word, you send letters one at a time; the board draws as you miss. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns. Add 'daily' for the Word of the Day — same word for everyone, all day.",
             "usage": "/hangman [daily]", "example": "/hangman daily",
             "related": "/game list · /game leaderboard hangman · /game quit"},
    "numberguess": {"what": "number guess — the house thinks of a number, you narrow the range. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/numberguess", "example": "/numberguess",
             "related": "/game list · /game leaderboard numberguess · /game quit"},
    "two_truths": {"what": "two truths and a lie — everyone posts three claims, the table votes for the lie. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/two_truths", "example": "/two_truths",
             "related": "/game list · /game leaderboard two_truths · /game quit"},
    "wyrr": {"what": "wyrr — the house poses, the table picks, everyone sees who's who. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/wyrr", "example": "/wyrr",
             "related": "/game list · /game leaderboard wyrr · /game quit"},
    "spy": {"what": "spy — everyone gets a word but one of you gets a fake; find the spy. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/spy", "example": "/spy",
             "related": "/game list · /game leaderboard spy · /game quit"},
    "auction": {"what": "auction — the table bids its points up for what the house puts on the block. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/auction", "example": "/auction",
             "related": "/game list · /game leaderboard auction · /game quit"},
    "trivia": {"what": "trivia royale — timed trivia rounds, the table races the clock and each other. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/trivia", "example": "/trivia",
             "related": "/game list · /game leaderboard trivia · /game quit"},
    "mafia": {"what": "mafia — town vs mafia with a night phase; the house runs the lynch and the kills. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/mafia", "example": "/mafia",
             "related": "/game list · /game leaderboard mafia · /game quit"},
    "king": {"what": "king of the hill — climb the hill and defend your spot from challengers. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/king", "example": "/king",
             "related": "/game list · /game leaderboard king · /game quit"},
    "story": {"what": "story chain — one sentence each, the story grows, the house judges the arc. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/story", "example": "/story",
             "related": "/game list · /game leaderboard story · /game quit"},
    "rpg": {"what": "rpg adventure — a shared dungeon crawl: the house sets the scene, the table decides. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/rpg", "example": "/rpg",
             "related": "/game list · /game leaderboard rpg · /game quit"},
    "shop": {"what": "shop game — the in-chat market turns into a game of spending and sniping. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/shop", "example": "/shop",
             "related": "/game list · /game leaderboard shop · /game quit"},
    "duel": {"what": "quiz duel — one-on-one rapid-fire questions, best of the round wins. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/duel", "example": "/duel",
             "related": "/game list · /game leaderboard duel · /game quit"},
    "pvp": {"what": "pvp — 1v1 combat against another human: your level, gear, and learned skills all fight. Start a lobby in a group "
                   "(/pvp), or challenge DM-to-DM (/pvp @user, /arena challenge @user). 120s per move; stall 3× and you forfeit.",
             "usage": "/pvp [@user]", "example": "/pvp @user",
             "related": "/game list · /game leaderboard pvp · /arena challenge @user"},
    "raid": {"what": "raid — team up against a raid boss: massive HP, cleaves the party every 3rd round, enrages at 30% HP. "
                   "Loot splits by damage dealt. Start it in a group; hunters join with /game join.",
             "usage": "/raid", "example": "/raid",
             "related": "/game list · /game leaderboard raid · /arena raid"},
    "case": {"what": "the case — a cooperative investigation: clues drop, the table reasons together. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/case", "example": "/case",
             "related": "/game list · /game leaderboard case · /game quit"},
    "world": {"what": "world — a continuing town that never ends, it just grows. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/world", "example": "/world",
             "related": "/game list · /game leaderboard world · /game quit"},
    "escape": {"what": "escape room — 4 locks, 3 strikes each, the table breaks out together. Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/escape", "example": "/escape",
             "related": "/game list · /game leaderboard escape · /game quit"},
    "political": {"what": "political — 3 elections: pledge, campaign, vote, be mayor (groups). Start it from any chat; while it's live, "
                   "plain messages are moves and the engine owns the turns.",
             "usage": "/political", "example": "/political",
             "related": "/game list · /game leaderboard political · /game quit"},
    "news": {"what": "fetch and summarize the news feeds.",
             "usage": "/news [run [n]|briefing|status]", "example": "/news run",
             "related": "/research · /pulse"},
    "translate": {"what": "translate anything: text you type, a message you reply to, or a link (you get the article translated, as a gist when it's long). News and briefings already arrive in your language automatically — this is the on-demand form. The ladder: a local model when this machine carries one, then the brain, then free provider endpoints, each health-tracked; if every rung is down you get the original text, plainly marked as untranslated.",
             "usage": "/translate [to <lang>] <text> · /translate <url> · /translate (as a reply) · /translate lang <code> · /translate status", "example": "/translate Bonjour le monde",
             "related": "/news · /pulse · /tts"},
    "tl": {"what": "alias of /translate.",
            "usage": "/tl [to <lang>] <text…> — same as /translate",
            "example": "/tl es Good morning",
            "related": "/translate"},
    "steal": {"what": "claim media into Devon's own GIF/sticker library: reply to a sticker, GIF, animation or video with /steal and she keeps it (tagged with any words you add), ready to fire back on the right moment. In groups she only collects where her bot systems are claimed (§33).",
              "usage": "/steal [tags] (as a reply to media)",
              "example": "/steal (replying to a sticker) funny dance",
              "related": "/gif · /sticker · /emoji"},
    "sticker": {"what": "the sticker maker: words or an emoji become a real 512×512 sticker; reply to an image to make it a sticker, or to a short video/GIF for an animated one.",
                "usage": "/sticker <text> · /sticker (as a reply to an image or video)",
                "example": "/sticker no wahala",
                "related": "/steal · /convert · /gif"},
    "convert": {"what": "media conversions on the replied file, riding the house ffmpeg organ: sticker→MP4, GIF→MP4, MP4→GIF, MP4→MP3, MP3→video (waveform), image→sticker, video→sticker. In a group, /convert with an amount is the currency exchange instead: games coins ⇄ that group's sparks at the week's floating rate.",
                "usage": "/convert <name> (as a reply to media) · /convert lists the names",
                "example": "/convert mp4-to-mp3",
                "related": "/sticker · /steal"},
    "gif": {"what": "a GIF for the moment, from Devon's own library first (it grows from steals and sends) and then her GIF providers, health-checked like every source.",
            "usage": "/gif <what you're feeling>",
            "example": "/gif celebration dance",
            "related": "/steal · /sticker · /emoji"},
    "emoji": {"what": "the FULL Unicode emoji table — every emoji that exists, named and keyworded, including rare ones some phones can't render. Search by meaning, or hand her an emoji and she'll name it (ZWJ sequences included). And what the table DOESN'T have, she creates FOR REAL: /emoji make <words> authors a custom emoji of her own — her art at 100×100, registered in her Telegram custom-emoji pack where the surface allows, living in /customemoji with a unicode fallback (a search no single emoji covers creates one too). Emoji are not stickers: /emoji sticker <words> is the sticker maker, named for what it makes.",
              "usage": "/emoji <words> · /emoji <an emoji> · /emoji make <words> · /emoji sticker <words>",
              "example": "/emoji make a cat riding a rocket",
              "related": "/customemoji · /emojiart · /react · /sticker"},
    "emojiart": {"what": "emoji art — pictures as TEXT: a photo becomes braille dot art (padded with braille blanks so the grid holds in chat), ASCII ramp art, or an emoji mosaic; she draws described subjects in dots; her own combo library serves kaomoji, combos and small art by topic, and grows by her drawing and your saves (never bulk-copied). Remix verbs reshape the last piece; the fancy styler restyles words in Unicode font variants.",
                 "usage": "/emojiart dot|ascii|mosaic (reply to a picture or paste its link) · draw <subject> · combo <topic> · fancy <style> <words> · remix invert|mirror|smaller|bigger|ascii|braille|mosaic · save <name>",
                 "example": "/emojiart draw a sleepy cat",
                 "related": "/emoji · /customemoji · /sticker"},
    "customemoji": {"what": "Devon's own custom emoji set: named entries with a unicode face everywhere, platform custom refs where a platform carries them, and sticker art. Curated by the owner (or group admins).",
                    "usage": "/customemoji list · add <name> <emoji> · ref <name> <platform> <ref> · show <name> · remove <name>",
                    "example": "/customemoji add devon 🥷",
                    "related": "/emoji · /sticker"},
    "react": {"what": "Devon reacts to the message you reply to — name an emoji or let her pick from the moment. Telegram bots use the standard reaction set; the userbot takes any emoji; on WhatsApp both bridge engines carry the verb (the reaction key is built from the bridge's own memory — a message it never saw refuses honestly), the act is Devon's on your personal account there: your standing stop halts it, and it lands in the personal journal. Reacting again with the same emoji removes it — WhatsApp's own toggle.",
              "usage": "/react [emoji] (as a reply)",
              "example": "/react 🔥",
              "related": "/emoji · /gif · /forward · /statusreact"},
    "joke": {"what": "a joke, fetched fresh from her joke sources (JokeAPI in safe mode, then the official-joke-api — health-checked like every source; a two-parter lands in one message, punchline on its own line). All sources down earns one honest line, never an invented joke passed off as fetched.",
              "usage": "/joke",
              "example": "/joke",
              "related": "/dadjoke · /progjoke · /chuck"},
    "dadjoke": {"what": "a dad joke from icanhazdadjoke, the dad-joke shelf itself.",
              "usage": "/dadjoke",
              "example": "/dadjoke",
              "related": "/joke · /progjoke"},
    "progjoke": {"what": "a programming joke — the Programming shelf of JokeAPI, safe mode on.",
              "usage": "/progjoke",
              "example": "/progjoke",
              "related": "/joke · /dadjoke"},
    "chuck": {"what": "a Chuck Norris fact from chucknorris.io. (The facts are legends; the source is real.)",
              "usage": "/chuck",
              "example": "/chuck",
              "related": "/fact · /catfact"},
    "fact": {"what": "a random fact from uselessfacts (the random shelf), health-checked.",
              "usage": "/fact",
              "example": "/fact",
              "related": "/fact2 · /catfact · /numfact"},
    "fact2": {"what": "the fact of the day — uselessfacts' today shelf: one fact, same all day, different from /fact's random draw.",
              "usage": "/fact2",
              "example": "/fact2",
              "related": "/fact · /history"},
    "catfact": {"what": "a cat fact from catfact.ninja.",
              "usage": "/catfact",
              "example": "/catfact",
              "related": "/cat · /fact"},
    "numfact": {"what": "a trivia fact about a number, from numbersapi — name a number or take a random one.",
              "usage": "/numfact [number]",
              "example": "/numfact 42",
              "related": "/fact · /calc"},
    "advice": {"what": "a slip of advice from adviceslip. Taking it is your call.",
              "usage": "/advice",
              "example": "/advice",
              "related": "/quote · /activity"},
    "quote": {"what": "a quote with its author, from her quote sources (dummyjson's quotable-class shelf).",
              "usage": "/quote",
              "example": "/quote",
              "related": "/advice · /wisdom"},
    "dog": {"what": "a dog picture, attached — from dog.ceo, breed named in the caption when the source's URL carries it. The bytes are sniffed as a real image before they ride; never a bare link.",
              "usage": "/dog",
              "example": "/dog",
              "related": "/cat · /fox"},
    "cat": {"what": "a cat picture, attached — thecatapi first, cataas as the second seed, health-checked; bytes sniffed before they ride.",
              "usage": "/cat",
              "example": "/cat",
              "related": "/dog · /catfact"},
    "fox": {"what": "a fox picture, attached — from randomfox.ca, bytes sniffed before they ride.",
              "usage": "/fox",
              "example": "/fox",
              "related": "/dog · /cat"},
    "meme": {"what": "a meme, attached with its title and subreddit in the caption — from meme-api. Flagged (nsfw/spoiler) hits are skipped and re-drawn: these play in groups.",
              "usage": "/meme",
              "example": "/meme",
              "related": "/joke · /gif"},
    "riddle": {"what": "a riddle game for the room: /riddle posts one (source chain first, her shipped bank of classics as the floor) and HOLDS the answer; /riddle answer reveals it. A fresh /riddle replaces the held one.",
              "usage": "/riddle · /riddle answer",
              "example": "/riddle",
              "related": "/trivia · /8ball"},
    "8ball": {"what": "the Magic 8-Ball — the canonical twenty answers, drawn fairly. Ask it a yes/no question first, for form.",
              "usage": "/8ball [question]",
              "example": "/8ball will it rain today",
              "related": "/choose · /riddle"},
    "choose": {"what": "Devon picks one of YOUR options — written the way people write them ('pizza or sushi', 'a, b, c'). One option isn't a choice and she'll say so (unless there's literally one: then that one wins).",
              "usage": "/choose <a> or <b>",
              "example": "/choose tea or coffee",
              "related": "/8ball"},
    "activity": {"what": "something to do when bored — from the Bored API (App Brewery's rehome first, the original as second seed), with its type and party size.",
              "usage": "/activity",
              "example": "/activity",
              "related": "/advice · /joke"},
    "pokemon": {"what": "a Pokémon record from PokéAPI, artwork attached: dex number, types, height, weight, base stats and a flavor line in the caption. Name one, number one, or take a random draw across all 1025. A name the Pokédex doesn't hold is answered honestly, never as an outage.",
              "usage": "/pokemon [name|number]",
              "example": "/pokemon pikachu",
              "related": "/nasa · /meme"},
    "history": {"what": "on this day — real events for the date from Wikimedia's on-this-day feed (MuffinLabs as second seed): five entries with their years. Name a date ('nov 5', '2026-11-05') or take today.",
              "usage": "/history [date]",
              "example": "/history nov 5",
              "related": "/fact2 · /news"},
    "nasa": {"what": "NASA's astronomy picture of the day, attached, with its real explanation trimmed to caption length. Video days deliver the video's still and say so; DEMO_KEY rate limits are named as rate limits when they bite (set NM_NASA_API_KEY for a bigger quota).",
              "usage": "/nasa",
              "example": "/nasa",
              "related": "/pokemon · /image"},
    "calc": {"what": "arithmetic, computed by the games organ's own checked evaluator (the path /math verifies with) — numbers with + − × ÷, brackets, % and powers. Not a code eval: names and calls are refused honestly.",
              "usage": "/calc <expression>",
              "example": "/calc 12 * (3 + 4) / 2",
              "related": "/math · /numfact"},
    "qcard": {"what": "a quote CARD: the words rendered as a polished chat-style card image (name, initial medallion, the text) and attached — made by her card renderer, openly a card: no forged timestamps or read ticks. Styles: quote · ttc · chat · ff · meme · duo. (/quote stays the text-quote command; this is the maker.)",
              "usage": "/qcard [style] <name> | <text> · /qcard (as a reply to a message)",
              "example": "/qcard Ada | we ship at dawn",
              "related": "/fakemsg · /sticker · /meme2"},
    "canvasquote": {"what": "the quote-card maker in its census skin — same renderer as /qcard, quote style.",
              "usage": "/canvasquote <name> | <text> · /canvasquote (as a reply)",
              "example": "/canvasquote Ada | we ship at dawn",
              "related": "/qcard · /iqc"},
    "iqc": {"what": "the quote-card maker in its census skin — same renderer as /qcard, quote style.",
              "usage": "/iqc <name> | <text> · /iqc (as a reply)",
              "example": "/iqc Ada | we ship at dawn",
              "related": "/qcard · /canvasquote"},
    "wndwqute": {"what": "the quote-card maker in its census skin — same renderer as /qcard, quote style.",
              "usage": "/wndwqute <name> | <text> · /wndwqute (as a reply)",
              "example": "/wndwqute Ada | we ship at dawn",
              "related": "/qcard · /iqc"},
    "meme2": {"what": "the card maker in meme style: the words as a big bold caption card, attached. Same renderer as /qcard; /meme stays the fetched-meme command.",
              "usage": "/meme2 <name> | <text> · /meme2 (as a reply)",
              "example": "/meme2 Ada | it works on my machine",
              "related": "/qcard · /meme"},
    "fakemsg": {"what": "the fake-message card maker: a styled message card (an openly made card, never a forged screenshot). Skins are styles of the one renderer: ttc (comment), chat (SMS), ff (game profile), duo (two speakers), quote. The census names are the same maker with the style built in.",
              "usage": "/fakemsg <style> <name> | <text> · /fakemsg duo <name> | <text> | <name> | <text>",
              "example": "/fakemsg ttc Ada | first!",
              "related": "/qcard · /fakettc · /fakeffduo"},
    "fakettc": {"what": "the card maker in TikTok-comment style — same renderer as /fakemsg with the style built in.",
              "usage": "/fakettc <name> | <text> · /fakettc (as a reply)",
              "example": "/fakettc Ada | this is the one",
              "related": "/fakemsg · /qcard"},
    "nokiamsg": {"what": "the card maker in SMS-chat style — same renderer as /fakemsg with the style built in.",
              "usage": "/nokiamsg <name> | <text> · /nokiamsg (as a reply)",
              "example": "/nokiamsg Ada | come home early",
              "related": "/fakemsg · /qcard"},
    "fakeff": {"what": "the card maker in game-profile style (navy + gold, the name as a player tag, the words as the status line) — same renderer as /fakemsg with the style built in.",
              "usage": "/fakeff <name> | <text> · /fakeff (as a reply)",
              "example": "/fakeff ShadowFox | booyah tonight",
              "related": "/fakemsg · /ffgirl"},
    "fakeffduo": {"what": "a two-speaker exchange card: two bubbles, two names — same renderer as /fakemsg in duo style.",
              "usage": "/fakeffduo <name> | <text> | <name> | <text>",
              "example": "/fakeffduo Ada | push mid | Bola | on my way",
              "related": "/fakemsg · /qcard"},
    "ffgirl": {"what": "the card maker in game-profile style — same renderer and style as /fakeff, in its census skin.",
              "usage": "/ffgirl <name> | <text> · /ffgirl (as a reply)",
              "example": "/ffgirl Luna | clutch or kick",
              "related": "/fakeff · /fakemsg"},
    "removebg": {"what": "background removal: the picture comes back as a transparent PNG, cut by the segmentation organ (rembg) — she self-provisions the model rung when it's missing and says so honestly when it can't land; never a silent chroma-key downgrade.",
              "usage": "/removebg (as a reply to a photo, or as the photo's caption)",
              "example": "/removebg",
              "related": "/image · /sticker · /convert"},
    "qr": {"what": "QR codes both ways: /qr <text> renders a scannable QR PNG and attaches it; /qr read on a replied picture decodes it with the vision organ's real decoder (pyzbar) — when that rung isn't provisioned the reply names it instead of guessing.",
              "usage": "/qr <text or link> · /qr read (as a reply to a picture)",
              "example": "/qr https://example.com",
              "related": "/shorturl · /lens · /decode"},
    "shorturl": {"what": "shorten a link through keyless shortener seeds (is.gd → v.gd → tinyurl, health-checked). The reply always shows the full destination next to the short link — a short link never hides where it goes.",
              "usage": "/shorturl <url>",
              "example": "/shorturl https://example.com/a/very/long/path",
              "related": "/unshorten · /qr"},
    "unshorten": {"what": "follow a link's redirects WITHOUT downloading what it points at, and name the page it finally lands on (and how many hops it took). A link that doesn't redirect is reported as pointing straight at its target.",
              "usage": "/unshorten <url>",
              "example": "/unshorten https://is.gd/q0jCDH",
              "related": "/shorturl · /qr"},
    "bible": {"what": "a bible lookup — not WisdomKeeper (that's corpus and practice): a passage quoted exactly from the source, reference labeled, source attributed. bible-api.com first, bolls.life behind it (and for search). KJV by default; end with a version (asv · web · darby · ylt · bbe) where the seed carries it. One chapter stretch per reply, verses capped.",
              "usage": "/bible <book> <chapter>[:<verse>[-<verse>]] [version] · /bible search <words>",
              "example": "/bible John 3:16",
              "related": "/wisdom · /quote · /search"},
    "couplepp": {"what": "a matched pair of profile pictures: ONE artwork from her anime picture pools, split into two complementary halves on one canvas — left half for one of you, right half for the other. Labeled as the single split artwork it is, never two unrelated pictures dressed as a pair.",
              "usage": "/couplepp",
              "example": "/couplepp",
              "related": "/pp · /waifu · /image"},
    "pp": {"what": "alias of /couplepp.",
              "usage": "/pp",
              "example": "/pp",
              "related": "/couplepp · /waifu"},
    "archive": {"what": "archive a chat on YOUR personal account (WhatsApp app-state / Telegram peer folders). The target is named the human way — a chat name, a number, or nothing for this chat; never a raw key.",
              "usage": "/archive [chat name]",
              "example": "/archive Family Group",
              "related": "/unarchive · /mutechat · /readchat"},
    "unarchive": {"what": "bring a chat back out of your archive, on your personal account.",
              "usage": "/unarchive [chat name]",
              "example": "/unarchive Family Group",
              "related": "/archive"},
    "mutechat": {"what": "mute a whole CHAT on your personal account (no end date) — the chat-level mute, not a per-member mute.",
              "usage": "/mutechat [chat name]",
              "example": "/mutechat",
              "related": "/unmutechat · /archive"},
    "unmutechat": {"what": "unmute a chat on your personal account.",
              "usage": "/unmutechat [chat name]",
              "example": "/unmutechat",
              "related": "/mutechat"},
    "readchat": {"what": "mark a whole chat read on your personal account.",
              "usage": "/readchat [chat name]",
              "example": "/readchat",
              "related": "/unreadchat"},
    "unreadchat": {"what": "mark a whole chat unread. Real on WhatsApp; Telegram honestly refuses — MTProto acknowledges reads and never un-acknowledges them.",
              "usage": "/unreadchat [chat name]",
              "example": "/unreadchat",
              "related": "/readchat"},
    "star": {"what": "star the message you reply to, on your personal WhatsApp account (the star key is built from the bridge's own memory). Telegram has no message star and says so.",
              "usage": "/star (as a reply to a message)",
              "example": "/star",
              "related": "/unstar"},
    "unstar": {"what": "remove the star from the message you reply to.",
              "usage": "/unstar (as a reply to a message)",
              "example": "/unstar",
              "related": "/star"},
    "forward": {"what": "forward the message you reply to into another chat, as you: the destination is named the way you name chats (a name, a number, or a full key) and must be on the same platform — a forward stays on its own platform. Both WhatsApp bridge engines re-send the ORIGINAL message out of bridge memory with the forwarded mark, never a retyped copy; a message the bridge never saw, and view-once messages, refuse honestly. One command is exactly one forward — the verb never retries. Telegram surfaces ride their native forward, attribution included. Owner only; your standing stop halts it; the act lands in the personal journal.",
              "usage": "/forward <chat name or number> (as a reply to the message)",
              "example": "/forward Gold Signals",
              "related": "/react · /poststatus · /chanstory"},
    "unsend": {"what": "delete YOUR OWN message for everyone, as a reply to it. WhatsApp enforces its ~60-hour window server-side and the refusal names it; editing or deleting other people's messages is never claimed.",
              "usage": "/unsend (as a reply to your message)",
              "example": "/unsend",
              "related": "/editmsg · /antiedit"},
    "edit": {"what": "alias of /editmsg.",
              "usage": "/edit <new text> (as a reply to your message)",
              "example": "/edit fixed the typo",
              "related": "/editmsg · /unsend"},
    "editmsg": {"what": "edit YOUR OWN message, as a reply to it with the new text. WhatsApp's ~15-minute window is enforced by WhatsApp; Telegram sets no window on your own messages. Other people's messages can't be edited anywhere.",
              "usage": "/editmsg <new text> (as a reply to your message)",
              "example": "/editmsg the meeting is at 5, not 4",
              "related": "/edit · /unsend · /antiedit"},
    "antiedit": {"what": "edit recovery, the antidelete organ's sibling: while it's on, originals of messages in your DMs are held, and when someone edits one the ORIGINAL is recoverable in /pair recovered — delivered to you, never called out in the room. Its toggle lives with the presence pack (/pair toggle) too.",
              "usage": "/antiedit [on|off]",
              "example": "/antiedit on",
              "related": "/pair · /unsend"},
    "block": {"what": "block a person on YOUR personal account — by number/@username, or by replying in your DM with them. WhatsApp rides the bridge blocklist verb; Telegram rides contacts.Block.",
              "usage": "/block <number or @username> · /block (as a reply in a DM)",
              "example": "/block +2348012345678",
              "related": "/unblock"},
    "unblock": {"what": "unblock a person on your personal account.",
              "usage": "/unblock <number or @username>",
              "example": "/unblock +2348012345678",
              "related": "/block"},
    "anticall": {"what": "while ON, incoming WhatsApp calls to your personal account are rejected as you (both bridge engines can see call offers and reject them). A presence-pack toggle — your standing stop halts it completely. Default off.",
              "usage": "/anticall [on|off]",
              "example": "/anticall on",
              "related": "/pair · /statusreact · /antitag"},
    "disappear": {"what": "set a chat's disappearing-messages timer on your personal account: off, 24h, 7d, 90d (WhatsApp's presets) or raw seconds. The Go bridge sets DMs and groups; the Baileys rung carries groups only and says so.",
              "usage": "/disappear <off|24h|7d|90d> [chat name]",
              "example": "/disappear 7d",
              "related": "/archive · /mutechat"},
    "statusreact": {"what": "while ON, Devon reacts to each contact's WhatsApp status once, as you, with your chosen emoji (default ❤️). A presence-pack toggle: default OFF, your standing stop halts it, and it never sends anyone a message — the reaction is its whole job.",
              "usage": "/statusreact [on|off] · /statusreact emoji <emoji>",
              "example": "/statusreact emoji 🔥",
              "related": "/anticall · /pair"},
    "viewonce": {"what": "see a view-once someone sent you. While the view_once toggle is on, every view-once photo/video that reaches your personal account is kept in your private vault before it can vanish; /viewonce sends the newest capture back into this chat as a normal photo/video you can actually open. Name a number to open an older one, or reply to the original message with it to open exactly that one. Owner only — the captures are yours alone.",
              "usage": "/viewonce · /viewonce <n> · /viewonce (as a reply to the view-once message) · /viewonce list",
              "example": "/viewonce",
              "related": "/vv · /pair saved · /pair toggle"},
    "vv": {"what": "short form of /viewonce — reveal a view-once capture from your private vault.",
              "usage": "/vv · /vv <n> · /vv (as a reply)",
              "example": "/vv",
              "related": "/viewonce · /pair saved"},
    "poststatus": {"what": "post a status as you, from your personal surface: on WhatsApp a text status, or an image/video status when the command rides a photo/video (as its caption, or as a reply to one); on Telegram a story from a photo/video the same way (stories are media posts — there is no text story). The post lands in the personal journal like every act taken as you. Owner only; your standing stop halts it.",
              "usage": "/poststatus [text] · send a photo/video with /poststatus [caption] as its caption · reply to a photo/video with /poststatus [caption]",
              "example": "/poststatus building while others flex",
              "related": "/statusreact · /pair saved · /pair"},
    "chanstory": {"what": "post to a channel as you, from your personal surface: on WhatsApp a channel post (text, or an image/video when the command rides a photo/video) — a WhatsApp channel's posts ARE its story format; on Telegram a channel story from a photo/video the same way (stories are media posts — there is no text story). The channel must be one your account admins: WhatsApp refuses a non-admin server-side, and Telegram also gates channel stories behind the channel's boost level — either refusal comes back in the platform's own words. Followed channels' posts are saved and marked viewed under the same toggles as statuses. The post lands in the personal journal like every act taken as you. Owner only; your standing stop halts it.",
              "usage": "/chanstory channel [text] · send a photo/video with /chanstory channel [caption] as its caption · reply to a photo/video with /chanstory channel [caption] — the channel as its link, JID, invite code or name on WhatsApp, its @name or id on Telegram",
              "example": "/chanstory Gold Signals building while others flex",
              "related": "/poststatus · /pair saved · /pair"},
    "joinlink": {"what": "join a WhatsApp group through its invite link, as you, from a personal surface: the bridge resolves the link first, so a dead link fails before any join attempt — a revoked link and a malformed one answer as the different facts they are. If the group files joins for admin approval, the reply says the request is in — it never claims you are a member before you are; the already-in case answers as itself and takes no action at all. A landed join (or filed request) lands in the personal journal like every act taken as you. Owner only; your standing stop halts it.",
              "usage": "/joinlink chat.whatsapp.com link or its code",
              "example": "/joinlink https://chat.whatsapp.com/AbC123xYz",
              "related": "/leave · /pair"},
    "newgroup": {"what": "create a Telegram group as you, from your personal Telegram surface (the userbot account — the bot surfaces answer where it lives). The group starts with you alone in it; add members with the member verbs afterwards. Telegram's own refusals come back in its words. The creation lands in the personal journal like every act taken as you. Owner only; your standing stop halts it. (WhatsApp group creation already lives in the whatsapp_group_create tool — this verb is the Telegram one.)",
              "usage": "/newgroup title",
              "example": "/newgroup Think Creativity Think Peace",
              "related": "/newchannel · /leave · /pair"},
    "newchannel": {"what": "create a Telegram channel as you, from your personal Telegram surface (the userbot account — the bot surfaces answer where it lives). Telegram caps how many channels one account may create; at the cap its refusal (CHANNELS_TOO_MUCH) comes back in its own words, never swallowed. The creation lands in the personal journal like every act taken as you. Owner only; your standing stop halts it.",
              "usage": "/newchannel title",
              "example": "/newchannel Devon Builds",
              "related": "/newgroup · /chanstory · /leave · /pair"},
    "leave": {"what": "leave a group or channel as you: run it inside the chat to leave that one, or name another chat the way you name chats (a name, a number, or a full key). WhatsApp groups ride the bridge's group_leave; Telegram groups and channels ride the userbot (the bot hat leaves through its own leaveChat when the target is a bot chat). A DM is not leavable and answers as exactly that. The leave lands in the personal journal like every act taken as you. Owner only; your standing stop halts it.",
              "usage": "/leave [chat name]",
              "example": "/leave",
              "related": "/joinlink · /newgroup · /newchannel · /pair"},
    "banchecker": {"what": "the honest ban check. With no argument: your own paired accounts' status and recorded bridge errors (a WhatsApp temporary ban shows up as a connection error there). With a number: whether it's on WhatsApp at all — the only signal WhatsApp publishes for other numbers; it publishes no ban state and this never pretends otherwise.",
              "usage": "/banchecker [phone number]",
              "example": "/banchecker +2348012345678",
              "related": "/pair"},
    "gcstatus": {"what": "this group's real stats, composed from data the surfaces actually hold — member counts, admin rosters, group settings — each labeled with its source. No platform offers a group-stats endpoint (the Bot API least of all), so nothing here is invented.",
              "usage": "/gcstatus (inside the group)",
              "example": "/gcstatus",
              "related": "/statusinfo · /pollres"},
    "statusinfo": {"what": "alias of /gcstatus.",
              "usage": "/statusinfo (inside the group)",
              "example": "/statusinfo",
              "related": "/gcstatus"},
    "pollres": {"what": "a poll's current results, re-read from the message itself, as a reply to it. Real on the personal Telegram surface (the tallies live on the poll media); the Bot API has no re-read endpoint and WhatsApp exposes none to the bridge — both say so.",
              "usage": "/pollres (as a reply to a poll)",
              "example": "/pollres",
              "related": "/poll · /gcstatus"},
    "tagall": {"what": "mention every member of this group by name, from your personal surface (the only one that can enumerate a room). Paced in chunks of 50 and capped at 100 members — the cap is named in the reply when a group is bigger.",
              "usage": "/tagall [text]",
              "example": "/tagall meeting in 10",
              "related": "/hidetag · /antitag"},
    "hidetag": {"what": "mention every member of this group WITHOUT visible names — the mention payload rides hidden (WhatsApp mention JIDs / Telegram zero-width mention entities), so the text reads clean while everyone is still notified. Same caps as /tagall.",
              "usage": "/hidetag [text]",
              "example": "/hidetag meeting in 10",
              "related": "/tagall · /antitag"},
    "antitag": {"what": "while ON, a group message that mass-tags you (5+ mentions including yours) is flagged to your DM — at most one flag per chat per hour, with the suppressed count carried into the next flag. Never a callout in the room. A presence-pack toggle; default off.",
              "usage": "/antitag [on|off]",
              "example": "/antitag on",
              "related": "/tagall · /anticall · /pair"},
    "awoo": {"what": "a reaction GIF: the sender awoos at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/awoo [@someone] · /awoo (as a reply to their message)",
            "example": "/awoo",
            "related": "/gif · /react"},
    "baka": {"what": "a reaction GIF: the sender calls the person you name a baka. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/baka [@someone] · /baka (as a reply to their message)",
            "example": "/baka",
            "related": "/gif · /react"},
    "bite": {"what": "a reaction GIF: the sender bites the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/bite [@someone] · /bite (as a reply to their message)",
            "example": "/bite",
            "related": "/gif · /react"},
    "blush": {"what": "a reaction GIF: the sender blushes at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/blush [@someone] · /blush (as a reply to their message)",
            "example": "/blush",
            "related": "/gif · /react"},
    "bonk": {"what": "a reaction GIF: the sender bonks the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/bonk [@someone] · /bonk (as a reply to their message)",
            "example": "/bonk",
            "related": "/gif · /react"},
    "bored": {"what": "a reaction GIF: the sender yawns at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/bored [@someone] · /bored (as a reply to their message)",
            "example": "/bored",
            "related": "/gif · /react"},
    "bully": {"what": "a reaction GIF: the sender bullies the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/bully [@someone] · /bully (as a reply to their message)",
            "example": "/bully",
            "related": "/gif · /react"},
    "cringe": {"what": "a reaction GIF: the sender cringes at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/cringe [@someone] · /cringe (as a reply to their message)",
            "example": "/cringe",
            "related": "/gif · /react"},
    "cry": {"what": "a reaction GIF: the sender cries beside the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/cry [@someone] · /cry (as a reply to their message)",
            "example": "/cry",
            "related": "/gif · /react"},
    "cuddle": {"what": "a reaction GIF: the sender cuddles the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/cuddle [@someone] · /cuddle (as a reply to their message)",
            "example": "/cuddle",
            "related": "/gif · /react"},
    "dance": {"what": "a reaction GIF: the sender dances with the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/dance [@someone] · /dance (as a reply to their message)",
            "example": "/dance",
            "related": "/gif · /react"},
    "facepalm": {"what": "a reaction GIF: the sender facepalms at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/facepalm [@someone] · /facepalm (as a reply to their message)",
            "example": "/facepalm",
            "related": "/gif · /react"},
    "feed": {"what": "a reaction GIF: the sender feeds the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/feed [@someone] · /feed (as a reply to their message)",
            "example": "/feed",
            "related": "/gif · /react"},
    "glomp": {"what": "a reaction GIF: the sender glomps the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/glomp [@someone] · /glomp (as a reply to their message)",
            "example": "/glomp",
            "related": "/gif · /react"},
    "handhold": {"what": "a reaction GIF: the sender holds hands with the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/handhold [@someone] · /handhold (as a reply to their message)",
            "example": "/handhold",
            "related": "/gif · /react"},
    "handshake": {"what": "a reaction GIF: the sender shakes hands with the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/handshake [@someone] · /handshake (as a reply to their message)",
            "example": "/handshake",
            "related": "/gif · /react"},
    "happy": {"what": "a reaction GIF: the sender smiles at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/happy [@someone] · /happy (as a reply to their message)",
            "example": "/happy",
            "related": "/gif · /react"},
    "highfive": {"what": "a reaction GIF: the sender high-fives the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/highfive [@someone] · /highfive (as a reply to their message)",
            "example": "/highfive",
            "related": "/gif · /react"},
    "hug": {"what": "a reaction GIF: the sender hugs the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/hug [@someone] · /hug (as a reply to their message)",
            "example": "/hug",
            "related": "/gif · /react"},
    "kiss": {"what": "a reaction GIF: the sender kisses the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/kiss [@someone] · /kiss (as a reply to their message)",
            "example": "/kiss",
            "related": "/gif · /react"},
    "laugh": {"what": "a reaction GIF: the sender laughs at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/laugh [@someone] · /laugh (as a reply to their message)",
            "example": "/laugh",
            "related": "/gif · /react"},
    "lick": {"what": "a reaction GIF: the sender licks the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/lick [@someone] · /lick (as a reply to their message)",
            "example": "/lick",
            "related": "/gif · /react"},
    "nom": {"what": "a reaction GIF: the sender noms the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/nom [@someone] · /nom (as a reply to their message)",
            "example": "/nom",
            "related": "/gif · /react"},
    "pat": {"what": "a reaction GIF: the sender pats the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/pat [@someone] · /pat (as a reply to their message)",
            "example": "/pat",
            "related": "/gif · /react"},
    "poke": {"what": "a reaction GIF: the sender pokes the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/poke [@someone] · /poke (as a reply to their message)",
            "example": "/poke",
            "related": "/gif · /react"},
    "pout": {"what": "a reaction GIF: the sender pouts at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/pout [@someone] · /pout (as a reply to their message)",
            "example": "/pout",
            "related": "/gif · /react"},
    "shrug": {"what": "a reaction GIF: the sender shrugs at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/shrug [@someone] · /shrug (as a reply to their message)",
            "example": "/shrug",
            "related": "/gif · /react"},
    "slap": {"what": "a reaction GIF: the sender slaps the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/slap [@someone] · /slap (as a reply to their message)",
            "example": "/slap",
            "related": "/gif · /react"},
    "sleep": {"what": "a reaction GIF: the sender sleeps beside the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/sleep [@someone] · /sleep (as a reply to their message)",
            "example": "/sleep",
            "related": "/gif · /react"},
    "smile": {"what": "a reaction GIF: the sender smiles at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/smile [@someone] · /smile (as a reply to their message)",
            "example": "/smile",
            "related": "/gif · /react"},
    "smug": {"what": "a reaction GIF: the sender looks smug at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/smug [@someone] · /smug (as a reply to their message)",
            "example": "/smug",
            "related": "/gif · /react"},
    "stare": {"what": "a reaction GIF: the sender stares at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/stare [@someone] · /stare (as a reply to their message)",
            "example": "/stare",
            "related": "/gif · /react"},
    "thumbsup": {"what": "a reaction GIF: the sender gives the person you name a thumbs up. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/thumbsup [@someone] · /thumbsup (as a reply to their message)",
            "example": "/thumbsup",
            "related": "/gif · /react"},
    "tickle": {"what": "a reaction GIF: the sender tickles the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/tickle [@someone] · /tickle (as a reply to their message)",
            "example": "/tickle",
            "related": "/gif · /react"},
    "wave": {"what": "a reaction GIF: the sender waves at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/wave [@someone] · /wave (as a reply to their message)",
            "example": "/wave",
            "related": "/gif · /react"},
    "wink": {"what": "a reaction GIF: the sender winks at the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/wink [@someone] · /wink (as a reply to their message)",
            "example": "/wink",
            "related": "/gif · /react"},
    "yeet": {"what": "a reaction GIF: the sender yeets the person you name. Fetched fresh from her reaction sources (waifu.pics / nekos.best, health-checked; the bytes are sniffed as a real animation before they can ride), with her library as the floor. The caption rides the GIF itself and names who did it to whom; name nobody and Devon performs it herself.",
            "usage": "/yeet [@someone] · /yeet (as a reply to their message)",
            "example": "/yeet",
            "related": "/gif · /react"},
    "pulse": {"what": "briefings: morning, afternoon, evening, midnight — check the schedule or trigger any briefing on demand. Each briefing covers what's new since the last one; long briefings arrive in full, paginated.",
             "usage": "/pulse [status|config|run|catchup] [morning|afternoon|evening|midnight|all]", "example": "/pulse run midnight",
             "related": "/news · /brief"},
    # (no second "research" entry: the single entry above is the web-research page)
    "code": {"what": "the coding bot: drafts code, runs it in the sandbox, fixes until the acceptance command passes; the real file lands in her workspace.",
             "usage": "/code <what to build>", "example": "/code a todo cli in python",
             "related": "/py /devon /task"},
    "py": {"what": "run python in the sandbox; -s name keeps a session, -r resets it.",
           "usage": "/py <code> [-s name] [-r]", "example": "/py print(6*7)",
           "related": "/code"},
    "checkpoint": {"what": "snapshot the coding workdir + session state so a bad run can be rewound.",
           "usage": "/checkpoint [label]", "example": "/checkpoint before refactor",
           "related": "/rewind /checkpoints /code"},
    "rewind": {"what": "restore a checkpoint's files (HEAD never moves); a pre-rewind safety checkpoint is saved first.",
           "usage": "/rewind [n]", "example": "/rewind 1",
           "related": "/checkpoint /checkpoints"},
    "checkpoints": {"what": "list saved coding checkpoints, newest first.",
           "usage": "/checkpoints", "example": "/checkpoints",
           "related": "/checkpoint /rewind"},
    "remember": {"what": "store something in her long-term memory.",
                 "usage": "/remember <text> [kind] [tags:a,b]",
                 "example": "/remember mom's birthday is May 4 tags:people",
                 "related": "/recall /forget"},
    "review": {"what": "how well did I represent you? reviews my autonomous actions on your behalf — outcomes, quality scores, weak spots.",
                 "usage": "/review [days]  (also: /how did I do)",
                 "example": "/review 7",
                 "related": "/spending"},
    "tutor": {"what": "socratic tutoring — guide-me questions or just-tell-me teaching, with a mistake notebook.",
                 "usage": "/tutor [guide me] <topic> | /tutor answer <text> | /tutor hint | /tutor status | /tutor stop",
                 "example": "/tutor guide me fractions",
                 "related": "/review"},
    "tour": {"what": "smoke-test every command with safe read-only probes — finds broken commands without running anything destructive. /tour <command> deep-probes one.",
             "usage": "/tour | /tour <command>",
             "example": "/tour research",
             "related": "/help /status /redteam /heal"},
    "heal": {"what": "self-healing loop — finds broken commands (tour), diagnoses them (error doctor), auto-fixes safe patterns (unbound-variable init, dead-method rescue), verifies, and reports. Each fix is git-committed for reversibility.",
             "usage": "/heal | /heal --dry | /heal <command>",
             "example": "/heal --dry",
             "related": "/tour /help"},
    "update": {"what": "Self-update from chat: mission gate (refuses mid-mission), checkpoint resumable missions, transactional pull + migrate + health-check (auto-rollback on failure), then restart and resume. The chat layer over `nm update`.",
             "usage": "/update",
             "example": "/update",
             "related": "/pull /deploy /restart /mission"},
    "pull": {"what": "Pull the latest code (git pull --ff-only) without restarting. The running process keeps its loaded code; new code takes effect on the next restart. Refuses mid-mission.",
             "usage": "/pull",
             "example": "/pull",
             "related": "/update /deploy /restart"},
    "deploy": {"what": "Deploy this code: same as /update (gate + checkpoint + pull + restart).",
             "usage": "/deploy",
             "example": "/deploy",
             "related": "/update /pull /restart"},
    "restart": {"what": "Restart the process: mission gate, checkpoint resumable missions, restart, resume from checkpoints. No code pull — use /update for that.",
             "usage": "/restart",
             "example": "/restart",
             "related": "/update /mission"},
    "course": {"what": "WAEC/JAMB course builder — scoping questions, syllabus-mapped lessons, embedded tutor.",
                 "usage": "/course <topic> | /course waec <subject> | /course set <field> <value> | /course build | /course list | /course teach <id>",
                 "example": "/course quadratic equations",
                 "related": "/tutor"},
    "routine": {"what": "natural-language smart-home routines — \"every morning at 7, kitchen lights + coffee\" → validated automation. You confirm before it activates.",
               "usage": "/routine <natural language> | /routine confirm <id> | /routine list",
               "example": "/routine every morning at 7 turn on the kitchen lights",
               "related": "/schedule"},
    "home": {"what": "home digital twin — your home's persistent model: current state, what changed, anomalies. Statistical, not 'intelligent'.",
             "usage": "/home status | /home what changed [hours] | /home unusual?",
             "example": "/home unusual?",
             "related": "/routine"},
    "store": {"what": "self-hosted storefronts — Medusa (zero platform tax) or WooCommerce (budget). Provision, AI catalog, conversational management.",
              "usage": "/store provision <business> | /store woo <business> <url> | /store list | /store catalog <id> <desc> | /store <id> <instruction>",
              "example": "/store provision SuyaSpot",
              "related": "/routine"},
    "mandate": {"what": "payment mandates — the agent's standing authority to move money (scope, per-transaction and daily caps, expiry). Every money-moving dispatch checks the active mandate.",
              "usage": "/mandate issue <scope> <per-txn-₦> [per-day-₦] [days-valid] | /mandate list | /mandate revoke <id> | /mandate stop all",
              "example": "/mandate issue transfer 50000 200000 30",
              "related": "/spend"},
    "train": {"what": "recovery-gated training plans — text-to-structured-plan (12-week blocks), daily workouts gated by readiness (HRV/sleep/RHR), voice coaching cues. Never a hard session on a low-recovery day.",
              "usage": "/train plan <goal> [equipment] [weeks] | /train today | /train readiness | /train log [completed|skipped] [rpe] [notes] | /train list",
              "example": "/train plan strength full 12",
              "related": "/health"},
    "form": {"what": "Seer-powered form coach — video movement analysis (squat/deadlift/push-up/plank/lunge) with per-checkpoint fixes, running gait risk flags, and mobility work fed back into today's training.",
              "usage": "/form analyze <video-or-photo> <exercise> | /form gait <video> | /form movements | /form apply <analysis-id>",
              "example": "/form analyze /tmp/squat.mp4 squat",
              "related": "/train"},
    "film": {"what": "VOD film study — upload a match/gameplay video → 3 mistakes + 3 good plays with timestamps, coaching notes, and a longitudinal playstyle fingerprint.",
              "usage": "/film analyze <video> <game> [exercise] | /film games | /film fingerprint [game] | /film list [game] | /film export <id>",
              "example": "/film analyze /tmp/match.mp4 valorant",
              "related": "/form"},
    "audio": {"what": "Transcript-as-timeline audio editing — transcribe any voice note/meeting with word timings, then edit the text: deletions cut, rewrites re-speak in the speaker's cloned voice.",
              "usage": "/audio edit <file> [lang] | /audio fillers <file> [lang] | /audio enhance <file>",
              "example": "/audio fillers /tmp/voice_note.m4a",
              "related": "/podcast"},
    "overview": {"what": "Interactive Audio Overviews — turn research/notes into a two-host podcast you can interrupt and question. Grounded in your sources; refuses rather than confabulates.",
              "usage": "/overview make <deep-dive|debate|brief> [lang] <Title :: text ;; Title2 :: text2> | /overview list | /overview ask <question> | /overview voices",
              "example": "/overview make deep-dive yo-ekiti \"Lagos rents :: notes here\"",
              "related": "/audio"},
    "character": {"what": "Conversational story characters — interview the people in your books, in character, with a hard knowledge cutoff. Spoiler discipline: the character only knows up to their chapter cutoff, enforced in the system prompt.",
              "usage": "/character list | /character talk <name> <question> | /character add <name> [book] [--cutoff N] | /character cutoff <name> <chapter> | /character forget <name>",
              "example": "/character talk Renfield what do you want most?",
              "related": "/overview"},
    "audiobook": {"what": "EPUB → audiobook, one click — chaptered TTS in your cloned voice, LUFS-mastered, with AI-disclosure metadata attached per store (ACX, Spotify, Kobo).",
              "usage": "/audiobook make <epub> [narrator=<ref>] [chapter:2=<ref>] [stores...] | /audiobook status",
              "example": "/audiobook make mybook.epub narrator=voice.wav spotify kobo",
              "related": "/overview /audio"},
    "graph": {"what": "Live world-graph planning substrate — people, projects, commitments, flights and deadlines as nodes with depends_on/blocks edges. Disruptions propagate: cancel a flight and every dependent meeting gets flagged.",
              "usage": "/graph add <type> <label> | /graph link <from-id> <to-id> <type> | /graph disrupt <id> [note] | /graph show [id] | /graph breaks <id> | /graph path | /graph sync",
              "example": "/graph add schedule \"Lagos flight 14:00\"",
              "related": "/routine"},
    "route": {"what": "Predict → Build → Solve routing — \"I need to do these 5 things this afternoon\" → optimal order with honest arrival times. The cost model learns from your recorded trips.",
              "usage": "/route plan <stop1>; <stop2>; ... | /route add <label> [lat,lng] | /route stops | /route record <from> > <to> <min> [kobo] | /route stats",
              "example": "/route plan home; market; bank",
              "related": "/graph"},
    "eta": {"what": "Honest time estimates — every promise ships a band, not a point. Bands widen after misses; quoted deadlines are the conservative end.",
            "usage": "/eta <task-type> [seg=min] ... | /eta record <task> <predicted-min> <actual-min> | /eta risk <task> <elapsed-min> | /eta stats [task]",
            "example": "/eta research prep=10 read=20 write=15",
            "related": "/route"},
    "congestion": {"what": "Multi-agent congestion prediction — per-resource queue depth with jam forecasting before fan-out. Back off, switch provider, or stagger.",
            "usage": "/congestion status | /congestion register <name> [kind] [capacity] | /congestion advise <resource> [agents=N] | /congestion predict <resource> | /congestion alternate <resource> <alternate>",
            "example": "/congestion advise groq-key agents=6",
            "related": "/route"},
    "scamcheck": {"what": "Nigerian rental scam check — six automated checks on any listing: price anomaly, duplicates, reverse-image, illegal fees, vague address, payment channel.",
            "usage": "/scamcheck <jiji/propertypro link or listing text> [photo:<path>]",
            "example": "/scamcheck https://jiji.ng/...",
            "related": "/contract"},
    "passport": {"what": "Verify-once rental passport — reusable tenant profile: KYC summary, income band (never exact), rental history, references, vault document references. Owner only.",
            "usage": "/passport generate [name] | show [id] | income <band> | history add <address> | <landlord> | <years> | ref add <name> | <relationship> | <contact-ref-id> | doc add <vault-doc-id> | <label> | export [id]",
            "example": "/passport generate",
            "related": "/truecost /scamcheck"},
    "truecost": {"what": "True rental move-in cost — parses a listing's fee language, fills gaps with Lagos norms (agency 10%, legal 10%), shows headline vs real move-in. Owner only.",
            "usage": "/truecost <paste listing text> | /truecost afford <listing text>",
            "example": "/truecost 2br Yaba ₦2,200,000/yr",
            "related": "/passport /scamcheck"},
    "value": {"what": "DIY property valuation — value RANGE (not a point) with confidence band and comparables. Thin markets get honestly wide bands.",
            "usage": "/value <e.g. '2br flat in Surulere'> — then /valuepick <id> <1,2,4> to refine from the comps you trust",
            "example": "/value 2br Surulere",
            "related": "/scamcheck /truecost"},
    "valuepick": {"what": "Comp-picker — refine a value estimate from the comparables you keep. Redfin trick.",
            "usage": "/valuepick <estimate id> <comp numbers, e.g. 1,2,4>",
            "example": "/valuepick val_abc123 1,2,4",
            "related": "/value"},
    "watch": {"what": "Persistent match alerts — saved searches that push the moment a listing matches: 'tell me the moment a 2-bed under \u20a61.5M hits Yaba'. Property first; gig/flight inherit. Auto-expires (7/14/28d). Owner only.",
            "usage": "/watch 2bed Yaba under 1.5m [for 7|14|28 days] | /watch gig <query> | /watch flight <route> | /watch list | /watch stop <id> | /watch check",
            "example": "/watch 2bed Yaba under 1.5m",
            "related": "/track /scamcheck"},
    "aeo": {"what": "AEO/GEO visibility tracking — share of answer, not rank. Fans prompts to ChatGPT/Claude/Gemini/Perplexity, parses citations, two-model cross-checks mentions, reports confirmed share. Owner only.",
            "usage": "/aeo track <brand> [prompt1; prompt2; …] | /aeo report [brand] | /aeo briefs [brand]",
            "example": "/aeo track Acme",
            "related": "/research"},
    "brief": {"what": "SERP-derived content brief — keywords, questions people ask, competitor angles, outline. First step of the brief-first pipeline: no brief, no draft, no schedule. Owner only.",
            "usage": "/brief <topic> | /brief run <topic>",
            "example": "/brief nigerian landlords",
            "related": "/content"},
    "content": {"what": "Brief-first content pipeline — score drafts against their brief, predict engagement, schedule (refuses drafts with no brief attached). Owner only.",
            "usage": "/content score <draft> | /content schedule <draft_id> <when> | /content run <topic>",
            "example": "/content score Lagos rents are wild right now",
            "related": "/brief"},
    "guardrails": {"what": "Rule-based ad guardrails — autonomous-spend safety. 'pause if CPA > X for 3 days', 'scale 20% if ROAS > 3'. Spend actions need an adspend mandate; explicit overrides bypass rules but never the mandate. Audit-logged. Owner only.",
            "usage": "/guardrails add <rule> | list | run | override <action> <adset> [pct] | stop <id> | audit",
            "example": "/guardrails add pause if cpa > ₦50000 for 3 days on adset123",
            "related": "/mandate /send"},
    "cost": {"what": "Transparent spend display — 'this answer cost $0.003' after every reply (toggleable), today's metered spend, and research budget caps. Builds trust; the router already meters every call. Owner only.",
            "usage": "/cost [on|off] | /cost budget $0.50",
            "example": "/cost on",
            "related": "/research"},
    "competitor": {"what": "No-access competitor intel — benchmark any public account without logins or credentials. Content pillars, posting cadence by format, engagement trends, window-vs-window digests, plus an AEO share-of-answer face-off against your brand. Owner only.",
            "usage": "/competitor track <account> [platform] | report <account> [platform] | digest [account] | list | untrack <account> | vs <competitor> <your-brand>",
            "example": "/competitor report rivalbrand",
            "related": "/aeo /brief"},
    "send": {"what": "Self-hosted send layer — Devon owns the mailgun. SQLite queue, per-provider sliding-window rate limits, exponential-backoff retry, DSN bounce processing with auto-blocklist, conditional templates, provider registry (SMTP/SES/Twilio/custom). Every bulk send routes through #68 cost-awareness. Owner only.",
            "usage": "/send template add <name> | <body> | /send provider add <name> <kind> [rate/min] | /send campaign <template> to <a@b.com, c@d.com> [via <provider>] | /send process | /send queue | /send status | /send bounce <address>",
            "example": "/send campaign launch to a@x.com, b@y.com via ses-main",
            "related": "/aeo /watch"},
    "leads": {"what": "Comment → DM → lead social-sales loop (ManyChat pattern). Keyword triggers on post comments fire auto-DM flows → lead capture → CRM → follow-up. WhatsApp legs priced through #68 cost-awareness; explicit 'DM everyone' commands always execute. Owner only.",
            "usage": "/leads trigger <post_id> <KEYWORD> [platform] | /leads triggers | /leads list [status] | /leads dm <contact> | /leads everyone <post_id>",
            "example": "/leads trigger post123 PRICE instagram",
            "related": "/send /content /aeo"},
    "track": {"what": "flight price watchers — 'track this for me' as persistent monitoring. Daily Duffel checks, owner-DM alert on drops with [Book] [Dismiss].",
              "usage": "/track LOS LHR 2026-12-01 [under 400k] | /track list",
              "example": "/track LOS LHR 2026-12-01 under 400k",
              "related": "/untrack"},
    "untrack": {"what": "stop a flight price watcher.",
              "usage": "/untrack <watch_id>",
              "example": "/untrack watch_abc123",
              "related": "/track"},
    "trip": {"what": "auto itineraries — forward a booking confirmation (email, PDF, screenshot) and get a structured trip with times, PNRs, calendar sync, and a document vault.",
              "usage": "/trip | /trip <id> | /trip add <text> | /trip calendar <id> | /trip docs <id>",
              "example": "/trip add Booking confirmation: BA075 LOS→LHR departs 22:45 PNR ABC123",
              "related": "/track"},
    "travelclient": {"what": "white-label travel assistants — per-client bots with isolated knowledge and shared travel infra (Duffel, watchers, itineraries). VIKI Nigeria template included.",
              "usage": "/travelclient create <name> | /travelclient knowledge <id> <title> | <text> | /travelclient ask <id> <q> | /travelclient spend <id> | /travelclient list | /travelclient viki [airline]",
              "example": "/travelclient viki ValueJet",
              "related": "/track"},
    "health": {"what": "patient-side health timeline — your own log of symptoms, visits, meds, measurements. Tracking only; not medical advice. Plus biometric coaching: /health ask, /health readiness, /health week (HealthKit/Health Connect data).",
               "usage": "/health log <text> | /health timeline | /health summary [days] | /health route <symptoms> | /health costs | /health prep [symptoms] | /health visited <notes>",
               "example": "/health route headache and fever since morning",
               "related": "/tutor"},
    "ekiti": {"what": "Ekiti/Ilawe Ekiti dialect tutoring — conversation, debate, pronunciation, reference audio.",
                 "usage": '/ekiti converse <text> | /ekiti debate <topic> | /ekiti say <text> | /ekiti check "<expected>" | /ekiti tones',
                 "example": "/ekiti debate school",
                 "related": "/tutor"},
    "recall": {"what": "what she remembers, ranked; a query narrows it.",
               "usage": "/recall [query]", "example": "/recall python",
               "related": "/remember /forget"},
    "forget": {"what": "delete a memory by id or description.",
               "usage": "/forget <id or description>", "example": "/forget mom's birthday",
               "related": "/recall"},
    "memories": {"what": "trust view: how many people she tracks, how many long-term memories she holds, recent contact. A name narrows to that person.",
               "usage": "/memories [name]", "example": "/memories Adaeze",
               "related": "/recall /remember"},
    "spend": {"what": "log a spend in Naira. Nigerian shorthand works: 5k, 2.5m.",
              "usage": "/spend <amount> [on <category>] [note...]",
              "example": "/spend 5k on transport",
              "related": "/spending /budget"},
    "budget": {"what": "monthly spending budgets per category. She warns at 80%, flags at 100%.",
               "usage": "/budget set <category> <amount> | /budget list",
               "example": "/budget set food 50k",
               "related": "/spend /spending"},
    "spending": {"what": "spending summary vs your budgets, with a weekly digest view.",
                 "usage": "/spending [week|month]", "example": "/spending week",
                 "related": "/spend /budget"},
    "tts": {"what": "speak text aloud and send it as a voice note (file=on sends the raw file instead). Pacing: speed=0.25-4.0, pace=tight|natural|slow, gap=<ms> (espeak-ng word gap). Tags: [pause] / [pause:300] insert real silence; [laugh] [cough] etc. are dropped (this engine cannot laugh — use /speak for real laughs); emotion tags stripped. Auto-tagging is ON by default (auto=off to disable): the tagger inserts engine-appropriate tags from plain-text cues (hesitation pauses everywhere; laughter/sentence/comma pauses where the engine renders them); explicit tags are always preserved, never duplicated. /tts engines shows which engine is in use and what tags it supports.",
            "usage": "/tts [speed=1.0] [pace=natural] [gap=0] [auto=on] [file=off] <text> | /tts engines", "example": "/tts pace=slow gap=30 Hello. [pause:500] How are you?",
            "related": "/speak /stt /say"},
    "sing": {"what": "sing a melody with neural singing voice.",
             "usage": "/sing C4:0.5:hello D4:0.5:world",
             "example": "/sing C4:0.5:la E4:0.5:la G4:1:laaa",
             "related": "/tts /say"},
    # (no second "say" entry: the single entry above covers all three forms)
    "stt": {"what": "transcribe an audio file to text.",
            "usage": "/stt <path>", "example": "/stt /sdcard/audio/voice.ogg",
            "related": "/tts"},
    "look": {"what": "screen-reader analysis of a screenshot or image URL (she actually sees the pixels).",
             "usage": "/look <path|url> [focus]", "example": "/look /sdcard/pic.jpg what app is this",
             "related": "/image /lens"},
    "schedule": {"what": "cron-style jobs that run in-process: at / every / daily.",
                 "usage": "/schedule add <name> <when> <action> | list | rm | enable | disable | run | status",
                 "example": "/schedule add backup daily 02:00 backup",
                 "related": "/task"},
    "db": {"what": "inspect the database: tables, schema, counts, or a query.",
           "usage": "/db tables | schema <t> | query <sql> | counts",
           "example": "/db counts", "related": ""},
    "artifact": {"what": "saved artifacts — agent outputs kept in the artifact store: list them, inspect one, or send its actual file to this chat.",
                 "usage": "/artifact list | info <id|alias|uri> | send <id|alias|uri>",
                 "example": "/artifact send latest-report",
                 "related": "/db /file"},
    "api": {"what": "external API connectors: weather, fx, github, …",
            "usage": "/api list | <connector> [json]", "example": "/api weather city=Lagos",
            "related": ""},
    "connectors": {"what": "manage the service connectors (connect, status): the same adapters `nm connectors` manages, from chat.",
            "usage": "/connectors [list] | /connectors status <name> | /connectors connect <name>",
            "example": "/connectors connect leonardo",
            "related": "/api /notion /gcal /trello"},
    "provision": {"what": "self-provisioning: when a feature dies for a missing API key, Devon offers to fetch one herself — logs in (or creates the account, your identity, one per service), creates the key, validates it, vaults it and retries. Owner only; never offered on the public bot.",
            "usage": "/provision [service] | /provision login <service> <username>",
            "example": "/provision hf",
            "related": "/connectors /browser watch"},
    "pair": {"what": "pairing-with-code account pool (§15): pair your own WhatsApp accounts into Devon with an 8-character code — no QR — and Telegram accounts by phone + login code, all in one registry. Each account gets its own supervised session; per-account routing names a default; personal surfaces (status saving, view-once preservation, DM antidelete) toggle per account. The pool is the reach behind the group/channel tools. Owner only.",
            "usage": "/pair | /pair whatsapp <number> [label] | /pair telegram <number> [label] | /pair tgcode <id> <code> | /pair tgpass <id> <password> | /pair code <id> | /pair start|stop|restart|remove <id> | /pair use <id> | /pair route [platform] | /pair toggle <id> <status_saving|view_once|antidelete|always_online|status_viewing> <on|off> | /pair saved [id] | /pair recovered [id] | /pair page",
            "example": "/pair whatsapp 2348012345678 main",
            "related": "/provision /connectors"},
    "sudo": {"what": "the sudo rung of the owner system (§19 + §20): a sudo holder is recognized on every surface, exempt from automatic actions, and never the owner. A top-level command because the owner reaches for it directly — /sudo on its own answers who holds sudo in plain words; add/remove delegate to the /owner grammar, so there is one edit path and one held audit. Owner only.",
            "usage": "/sudo · /sudo add <platform> <id> · /sudo remove <platform> <id>",
            "example": "/sudo",
            "related": "/owner /owner status"},
    "owner": {"what": "the owner system (§19 + §20): one record binds the owner's platform identities — every surface resolves who the owner is from it, never from display names. The ladder is owner → sudo (recognized everywhere, exempt from automatic actions, never ownership) → group-scoped deputies; every edit lands in the record's held audit. The card shows each platform's bound ids, how they were proved (config seed, the §15 pool, the seal), and the seal's state; status carries uptime (it lives here, not on /ping), the version, the pool counts, queue depths and the last briefing. If the owner's platform id ever changes, a sealed DM rebind (identity claim + passphrase against the core seal) rewrites the binding. The record is never editable from a group, by anyone. Owner only.",
            "usage": "/owner · /owner status · /owner audit · /owner bind|unbind <platform> <id> · /owner sudo add|remove <platform> <id> · /owner deputy add|remove <group key> <id> · /owner mark <text> · /owner rebind <platform> <new id> <seal passphrase>",
            "example": "/owner status",
            "related": "/pair /console /gsettings"},
    "massreact": {"what": "mass react (§14): place one emoji reaction on a post from every connected account in your pairing pool at once — Telegram posts via your MTProto sessions, WhatsApp channel posts via each account's bridge. Paced with jitter, journaled per account, and arm-gated: the tool must be armed once (its warning is shown then), and its first run is a dry-run unless you say live. Your own accounts only — never the bot.",
            "usage": "/massreact <post link> | <emoji> [| count] [| live|dry] · /massreact arm|disarm|status|pacing <s>|runs",
            "example": "/massreact https://t.me/somechannel/482 | 🔥",
            "related": "/pollvote /chfollow /pair"},
    "pollvote": {"what": "mass vote (§14): vote in an external poll from every connected Telegram account in your pairing pool — MTProto only, because the Bot API cannot vote in polls at all. The option resolves by number or by its exact text against the poll's own answers before anything is cast. WhatsApp poll voting has no Baileys verb (protocol-level only) and honestly refuses there. Arm-gated like /massreact; first run is a dry-run unless you say live.",
            "usage": "/pollvote <poll link> | <option text or number> [| count] [| live|dry] · /pollvote arm|disarm|status|pacing <s>|runs",
            "example": "/pollvote https://t.me/somechannel/511 | 2",
            "related": "/massreact /chfollow /pair"},
    "chfollow": {"what": "channel follow (§14): every connected account in your pairing pool follows/subscribes to a channel — Telegram channels via MTProto join, WhatsApp channels via each account's bridge. Idempotent: an account that already follows counts as done. Arm-gated like /massreact; first run is a dry-run unless you say live.",
            "usage": "/chfollow <channel link> [| count] [| live|dry] · /chfollow arm|disarm|status|pacing <s>|runs",
            "example": "/chfollow https://whatsapp.com/channel/0029VaExample",
            "related": "/massreact /pollvote /pair"},
    "rules": {"what": "this group's rules — rendered live from the same settings the moderation enforcer reads, so what you see is what's enforced.",
            "usage": "/rules",
            "example": "/rules",
            "related": "/gsettings /member"},
    "member": {"what": "a group member's card: when they joined, how active they are, and their username/name changes as Devon noticed them. No argument = your own card.",
            "usage": "/member [@name]",
            "example": "/member @ada",
            "related": "/rules /glog"},
    "glog": {"what": "the group's audit log — moderation actions, joins, identity changes — held by Devon and shown on demand to group admins. Never posted on its own.",
            "usage": "/glog [n]",
            "example": "/glog 10",
            "related": "/gsettings /member"},
    "strikes": {"what": "the defense strike ledger: every sensor and every manual warn lands on one per-member ladder (mute at the warn limit, removal and ban above it, 7-day window). No argument lists who's on the ladder; a member shows their record.",
            "usage": "/strikes [@name]",
            "example": "/strikes @ada",
            "related": "/glog /gsettings /warn"},
    "recheck": {"what": "check me again: if the group's acceptance rules (§28) let you in restricted — muted until you fix what's missing, like setting a username — tap this (or the button on your fix-it note) after fixing it. The criteria re-run live; a pass lifts the restriction and the welcome fires.",
            "usage": "/recheck",
            "example": "/recheck",
            "related": "/joinq /gsettings /rules"},
    "joinq": {"what": "the admin join queue (§28): join requests the rules held for a human, plus members held by the raid posture. Approve lets them in (restricted, if their check said so), decline refuses the request or removes the held member; 'all' decides the whole queue at once. The notice on each request carries the same decisions as buttons. Where Telegram handed the bot a join query, holding a request in the queue is itself the answer ('queue' — the decision passes to the chat's other admins). 'ask' messages an applicant while their DM window is open (Telegram: 5 minutes from the request, if no other admin wrote first; WhatsApp: while their request stands).",
            "usage": "/joinq | /joinq approve <id> | /joinq decline <id> | /joinq approve all | /joinq ask <id> <message>",
            "example": "/joinq ask jr-0003 where did you hear about us?",
            "related": "/gsettings /glog /recheck"},
    "console": {"what": "the DM admin console (§30): message Devon privately and she lists the groups where YOU hold admin — pick one and configure its systems there in tappable sections (defense + the warn ladder, acceptance, economy + governance, estate levers, presentation, welcome/rules), with runtime status (uptime, per-platform connection, userbot session) on the home card. Changes apply at once and land in the group's /glog with your name on them. Owner-only switches stay owner-only, judged in the target group.",
            "usage": "/console | /console g1 | /console g1 defense escalation limit 5",
            "example": "/console",
            "related": "/gsettings /glog /strikes"},
    "undonotice": {"what": "undo one automated action from its notice (§31): the strike it recorded is removed, a fine is refunded, and the platform action is reversed as far as the platform allows (mute lifted, ban lifted, restriction released; a removal needs nothing — the member can rejoin; a deleted message can't be restored and the undo says so). The notice edits in place where the platform supports it. Admins only; the undo is journaled in /glog like the action.",
            "usage": "/undonotice <notice id> (or tap the notice's Undo button)",
            "example": "/undonotice n1730…-0",
            "related": "/glog /strikes /console"},
    "flag": {"what": "cross-group reputation (§32): stage a record on a member — scam, fraud, raid, harassment, spam-ring or other, with a short evidence note. Nothing is recorded until an admin confirms (the command answers with Confirm/Discard buttons); reports and defense-sensor bans propose flags into the same queue, and unconfirmed proposals expire in 3 days branding nobody. A confirmed record binds the resolved person — platform ids plus username history, so a changed handle doesn't shed it — and warns every group they join, per that group's warning mode. It never decides admission. 'retract' clears the records THIS group sourced, everywhere. Admins only.",
            "usage": "/flag @member <category> [note] | /flag | /flag ok <id> | /flag no <id> | /flag retract @member",
            "example": "/flag @ada scam took deposits for a fake signal room",
            "related": "/rep /repdismiss /repclear /glog"},
    "rep": {"what": "the reputation book (§32): one ledger across every group and platform Devon serves. With a member: their whole file — live records (category, source group, date, confirming admin, evidence note, where it has warned), plus retracted and aged history. Bare: the book's shape. A record never auto-rejects a join; §28 acceptance decides admission on the group's own criteria and the record only warns. Admins only.",
            "usage": "/rep [@member]",
            "example": "/rep @ada",
            "related": "/flag /repdismiss /member /glog"},
    "repclear": {"what": "lift a reputation restriction (§32 restricted mode): a flagged joiner admitted restricted stays that way until an admin clears them here (or taps Clear on the warning notice). The record itself stands in the book — clearing only ends this group's hold. Admins only.",
            "usage": "/repclear <member id> (or tap the notice's Clear button)",
            "example": "/repclear 424242",
            "related": "/rep /repdismiss /flag"},
    "repdismiss": {"what": "dismiss a reputation record for THIS group (§32): the group stops hearing about it — no warning on the next join, and any hold the record is causing here lifts. The record stands globally and still warns elsewhere; only the source group can retract it outright. Journaled in /glog and in the record's own history. Admins only.",
            "usage": "/repdismiss <record id> (or tap the notice's Dismiss button)",
            "example": "/repdismiss rep-0003",
            "related": "/rep /flag /repclear"},
    "gsettings": {"what": "this group's settings: welcome and goodbye switches, service-message cleanup, the rules list, flood limit, link allowlist, the economy switch and transfer tax, and the defense system (autonomy, anticrash, antichannel, antitag, antidelete, antiedit) with the platform limits stated plainly. Group admins only — Devon checks your platform admin status; the autonomy switch belongs to the group owner on Telegram.",
            "usage": "/gsettings | /gsettings welcome on|off | /gsettings rules add <text> | /gsettings rules clear | /gsettings flood <n> | /gsettings links <domains>|off | /gsettings economy on|off | /gsettings tax <pct> | /gsettings autonomy on|off | /gsettings anticrash on|off | /gsettings antitag <n>",
            "example": "/gsettings rules add No spam",
            "related": "/rules /glog /strikes"},
    "balance": {"what": "your sparks balance in this group, with your recent ledger and the group treasury. Sparks are group game points — earned by chatting, streaks, quests, work, mining and in-group game wins.",
            "usage": "/balance",
            "example": "/balance",
            "related": "/mine /work /quests /treasury"},
    "mine": {"what": "claim mined sparks. The rate decays with each claim in a day and claims cool down between hits — grinding is pointless by design; your hashrate is your real engagement this week.",
            "usage": "/mine",
            "example": "/mine",
            "related": "/balance /work"},
    "work": {"what": "work a shift for sparks: one shift an hour, pay follows your engagement (hashrate) and your shift level, which rises every 10 shifts.",
            "usage": "/work",
            "example": "/work",
            "related": "/balance /mine /quests"},
    "pay": {"what": "send sparks to another member of this group. A small tax (set by the admins, default 2%) goes to the group treasury.",
            "usage": "/pay @member <amount>",
            "example": "/pay @ada 50",
            "related": "/balance /treasury"},
    "gshop": {"what": "the group status shop: spend sparks on status — a shout-out, a pinned message, a custom sticker or emoji, a nickname change. Status, never power.",
            "usage": "/gshop | /gshop buy <item>",
            "example": "/gshop buy shout_out",
            "related": "/balance"},
    "quests": {"what": "the group's quests: daily and weekly goals (helpful answers, streaks, events, invites) with spark rewards, claimable once per period.",
            "usage": "/quests | /quests claim <id>",
            "example": "/quests claim helpful_3",
            "related": "/balance /work"},
    "treasury": {"what": "the group's treasury and economy health: what's in the pool, how much circulates, who holds it, how many members were active this week.",
            "usage": "/treasury",
            "example": "/treasury",
            "related": "/balance /pay /drop"},
    "drop": {"what": "start a giveaway drop (admins): you fund a pool from your own sparks; members claim a fixed share each with /claim until it runs dry. One claim per member.",
            "usage": "/drop <total> <per claim>",
            "example": "/drop 500 50",
            "related": "/claim /treasury"},
    "claim": {"what": "claim your share of the live giveaway drop.",
            "usage": "/claim",
            "example": "/claim",
            "related": "/drop /balance"},
    "deposit": {"what": "move sparks from your wallet into your bank vault. Banked sparks count toward your total wealth and the richest board, and robbers cannot touch them.",
            "usage": "/deposit 50 | /deposit all",
            "example": "/deposit all",
            "related": "/withdraw /balance /rob"},
    "withdraw": {"what": "move sparks back from your bank vault into your wallet (wallet sparks are what you spend, pay and gamble with).",
            "usage": "/withdraw 50 | /withdraw all",
            "example": "/withdraw 50",
            "related": "/deposit /balance"},
    "rob": {"what": "try to steal sparks from a member's WALLET (banked sparks are safe). Success takes 10–25% of their wallet; failure puts YOU in jail for 30 minutes — /bail pays the treasury to get out.",
            "usage": "/rob @member, or reply to their message with /rob",
            "example": "/rob @ada",
            "related": "/bail /deposit /balance"},
    "bail": {"what": "pay your bail to the group treasury and leave jail immediately (robbery failures land you there). Not jailed but in hospital? /bail pays the hospital bill instead and discharges you.",
            "usage": "/bail",
            "example": "/bail",
            "related": "/rob"},
    "gear": {"what": "your conflict loadout: the arsenal you carry (breach gear, weapons, armor, support), your street power and rep, and what guards your vault. Buy on the arsenal shelf in /gshop; install defenses on the treasury, a district bank or your own vault; use a medkit to leave hospital or a repair kit to restore a worn tool.",
            "usage": "/gear | /gear buy <item> | /gear install <item> treasury|bank <district>|vault | /gear use <item>",
            "example": "/gear install vault_plating vault",
            "related": "/gshop /heist /gang"},
    "heist": {"what": "a crew score: 2–5 members combine their gear against a target's installed defenses (the group treasury, a district bank, or a member's banked vault). The odds print before you run it — ratings, counter-pairs and police heat, never a bare coin flip. Success splits the take; partial pays less; a bust jails the whole crew. Big targets need a gang of 5+ behind the crew.",
            "usage": "/heist start treasury|bank <district>|vault @member · /heist join · /heist go",
            "example": "/heist start bank harbour",
            "related": "/gear /gang /rob"},
    "shoot": {"what": "the light hit verb: shoot a member and lift 5–10% of their wallet (cap 100). Your gear against their armor sets the odds, printed on the result. Failure: 10 minutes in jail.",
            "usage": "/shoot @member, or reply to their message with /shoot",
            "example": "/shoot @ada",
            "related": "/kill /rob /bail"},
    "kill": {"what": "the heavy hit verb: needs a real weapon (street pistol or better). Lifts 15–25% of the target's wallet (cap 250) and puts them in hospital for 20 minutes. Failure: 45 minutes in jail.",
            "usage": "/kill @member, or reply to their message with /kill",
            "example": "/kill @ada",
            "related": "/shoot /assassinate /gear"},
    "assassinate": {"what": "the contract kill: needs a heavy weapon (hunting rifle, strike drone or better). Lifts up to 28% of the target's wallet (cap 400) and puts them in hospital for 45 minutes. Failure: an hour in jail — and the police notice everything.",
            "usage": "/assassinate @member, or reply with /assassinate",
            "example": "/assassinate @ada",
            "related": "/kill /police /gear"},
    "sabotage": {"what": "wreck a member's best property: the estate organ bills them the damage through its own repair machinery — insurance answers first, then bank, wallet, repair debt. Failure: 20 minutes in jail.",
            "usage": "/sabotage @member, or reply with /sabotage",
            "example": "/sabotage @ada",
            "related": "/property /insure /heist"},
    "gang": {"what": "gang life: form or join a gang under a lifestyle (syndicate, cartel, bikers, street, mafia, vigilantes — each with its own income, gear affinities, war style and heat profile). Crews grow into gangs at 5 members and syndicates at 15. Gangs hold a treasury, an armory and territory; the strongest gang holding 2+ districts rules the group — its leaders drive treasury governance and set the transfer tax.",
            "usage": "/gang | /gang create <lifestyle> <name> | /gang join <name> | /gang job | /gang deposit <n> | /gang claim <district> | /gang tax <pct> | /gang list",
            "example": "/gang create cartel Night Runners",
            "related": "/war /gear /heist"},
    "war": {"what": "gang wars: declare on a gang here or in another claimed group; the other side matches the escrow from its gang treasury and accepts; members opt in, rounds fight on the group's cycle with gear and strength deciding. Winner takes the pot — and at home, one of the loser's districts as spoils.",
            "usage": "/war | /war declare <gang> | /war accept|decline | /war join | /war spoils <district>",
            "example": "/war declare Night Runners",
            "related": "/gang /gear"},
    "police": {"what": "the precinct board: the wanted list (driven by the bully index — farming the weak, the new and the same victim, heist heat, war stomps, community reports), open bounties, and your own heat. At the star thresholds the police fine, raid vaults, post bounties and jail. /police bribe buys heat off — a sink with real risk.",
            "usage": "/police | /police bribe [envelope]",
            "example": "/police bribe",
            "related": "/bounty /glog /bail"},
    "bounty": {"what": "put your own sparks on a member's head (escrowed in the treasury while it stands), or hunt a wanted member down: defeat them with a geared hit or a /bounty hunt and the bounty pays you whole. Vigilante gangs earn street rep for claims.",
            "usage": "/bounty | /bounty post @member <amount> | /bounty hunt @member",
            "example": "/bounty post @ada 200",
            "related": "/police /shoot /kill"},
    "claimdrop": {"what": "claim a system drop (🪂): contents roll fresh every time — sparks, gems or gear — across 1–5 first-come slots set by the group's daily drop budget. Claims are settlements: final and journaled, no undo.",
            "usage": "/claimdrop (or tap the drop's Claim button)",
            "example": "/claimdrop",
            "related": "/gear /balance /drop"},
    "poll": {"what": "open a group poll (🗳️): one counting core, rendered as a button card where the platform carries buttons and as /vote numbers everywhere. Plain polls are opinion; a TREASURY poll is a vote on a real governance proposal — pass it with quorum and the grant executes through governance on close; fail it and nothing moves. Polls close on the message cycle at their deadline and post their results as a card.",
            "usage": "/poll <question> | <option> | <option> [| for 2h] [| flexible] · /poll treasury grant @member 500 · /poll close [N] · bare /poll lists open polls",
            "example": "/poll Best session to trade? | London | New York",
            "related": "/vote /treasury /approve"},
    "vote": {"what": "vote in an open poll: one vote per member, final unless the poll was opened as flexible (then changeable until close). The poll card's buttons run this same command.",
            "usage": "/vote <option number> (newest poll) · /vote <poll number> <option number>",
            "example": "/vote 2",
            "related": "/poll"},
    "milestones": {"what": "the group's memory book (📖): milestones recorded as they happen — member join anniversaries, message-count landmarks for the group and its members, economy records (richest balance, biggest heist), the first gang, the first war. Seeded only from facts the ledgers already prove; nothing is backdated.",
            "usage": "/milestones",
            "example": "/milestones",
            "related": "/contributions"},
    "contributions": {"what": "your §26 contribution file: substantive posts are the rankers' raw material; when the group's review is on, an ensemble (and borderline cases an admin) decides what counts. See your counted / waiting / not-counted items with the plain-word reasons, appeal a rejection once, or read the reviewed board with /contributions top.",
            "usage": "/contributions · /contributions top · /contributions appeal <number>",
            "example": "/contributions top",
            "related": "/milestones /top /member"},
    "boards": {"what": "the §14 leaderboards: one board organ over many standings — this group's sparks, gems, messages, REVIEWED contributions and gang strength; per-platform and cross-platform standings with people stitched across platforms so one person ranks once; per-game ELO/wins/points and whole-spine XP/coins/mastery. Boards are living surfaces: a scheduled publish stores the snapshot and later renders show movement (▲2 / ▼1 / — / new) against it. /top stays the quick group sparks board.",
            "usage": "/boards · /boards <name> — sparks, global:sparks, platform:telegram-bot:messages, game:chess:elo, games:xp",
            "example": "/boards global:sparks",
            "related": "/top /contributions /milestones"},
    "webshot": {"what": "render a web page and return the screenshot as a photo, captioned with the page title + URL (§18, expanded §20): one command carries the whole variant space — full-page or viewport, phone/tablet/desktop profiles, dark mode, wait-for-idle, a single-element clip, the print/PDF rung, and annotate (the vision organ reads the shot and answers with it). Reply to a message holding a link to shoot that link. In groups it is a safe-preview: the shot posts only in claimed rooms (§33). Page-watch turns it into a surveillance utility: a watched URL is re-shot on the cycle and posts only when the page actually changes. Shots are taken by her own headless Chromium first, with health-gated provider seeds behind it; when no rung can honor the request, the failure names every rung's reason — never a fake image. Generated shots carry the group's watermark mode (§19).",
            "usage": "/webshot [full] [phone|tablet|desktop] [dark] [idle] [pdf] [annotate] [clip <css selector>] <url> · /webshot watch <url> [every 2h] · /webshot watch list · /webshot watch remove <id>",
            "example": "/webshot full phone https://example.com",
            "related": "/look /browser /owner"},
    "coinflip": {"what": "flip a coin against the house at 1.9×: call heads or tails. Losses feed the treasury and the jackpot pot.",
            "usage": "/coinflip <bet> heads|tails",
            "example": "/coinflip 10 heads",
            "related": "/slots /balance"},
    "top": {"what": "the richest board: members ranked by total wealth — wallet plus banked sparks plus property value at the live market index.",
            "usage": "/top",
            "example": "/top",
            "related": "/balance /deposit /property"},
    "tax": {"what": "the group's monetary policy, stated plainly: transfer tax, moderation fine, earn caps, the governance threshold, and whether issuance is running hot enough that faucets are throttled.",
            "usage": "/tax",
            "example": "/tax",
            "related": "/treasury /gsettings"},
    "grant": {"what": "pay a member out of the group treasury (admins). Above the governance threshold (see /tax) the grant becomes a proposal a second admin must /approve. The 'gems' form — /grant gems @member 25 — gifts gems 💎, the premium purse, which moves ONLY through this governance road.",
            "usage": "/grant @member <amount> · /grant gems @member <amount>",
            "example": "/grant @ada 100",
            "related": "/approve /treasury /drop"},
    "property": {"what": "the property map and your portfolio: districts with their live market index and trend, every tier priced per district, your buildings with their rent standing, your gems, and your management rung. Property is an asset class inside the group economy — it counts toward the richest board.",
            "usage": "/property · /property manager (hire the next management rung)",
            "example": "/property",
            "related": "/buyproperty /rent /sellproperty"},
    "buyproperty": {"what": "buy a building from the system at the district's index price, paid to the treasury. Spark tiers (room → plaza) price in sparks; penthouse and tower are the premium shelf — gems only. Admins can buy PUBLIC property for the treasury ('public'), whose rent funds drops and prizes; large public buys go through governance like any treasury spend.",
            "usage": "/buyproperty <district> <tier> · /buyproperty public <district> <tier>",
            "example": "/buyproperty market_mile shop",
            "related": "/property /sellproperty /rent"},
    "sellproperty": {"what": "sell a property back to the system at the depreciated rate (the group's sell-back %, see /property), in the currency its tier is priced in. A listed, auctioned or mortgaged property must be freed first.",
            "usage": "/sellproperty <property id>",
            "example": "/sellproperty prop-a1b2",
            "related": "/property /listproperty"},
    "rent": {"what": "collect the rent standing on all your properties. Rent accrues one period per day per building (capped by your management rung) and pays in GEMS 💎 — the premium purse. Repair debt is offset from rent first. A manager (level 1+) collects automatically, even while you're jailed or away.",
            "usage": "/rent",
            "example": "/rent",
            "related": "/property /buyproperty /insure"},
    "listproperty": {"what": "the member property market: list one of your buildings at your ask (bare /listproperty shows the board), buy a listed building, or cancel your listing. The treasury takes the group's market cut of every sale, journaled apart from the seller's net. The rent clock follows the property through a sale.",
            "usage": "/listproperty · /listproperty <id> <ask> · /listproperty buy <listing id> · /listproperty cancel <id>",
            "example": "/listproperty prop-a1b2 1200",
            "related": "/property /auctions /sellproperty"},
    "auctions": {"what": "property auctions run through treasury escrow: the standing high bid is HELD in the treasury, outbids are refunded in full, and settlement pays the seller minus the market cut. A bid inside the last 2 minutes extends the close (anti-snipe). Bare /auctions shows the live board with bid buttons.",
            "usage": "/auctions · /auctions start <id> <start price> [hours] · /auctions bid <auction id> <amount>",
            "example": "/auctions bid auc-a1b2 900",
            "related": "/listproperty /property"},
    "mortgage": {"what": "borrow sparks from the treasury against one property's value (up to the group's LTV), at a flat stated interest per week. Instalments auto-debit your BANK vault weekly; paying by hand is always open. Miss the third weekly instalment and the property is repossessed into public stock, the remaining debt written off on the record.",
            "usage": "/mortgage · /mortgage borrow <property id> <amount> · /mortgage repay <mortgage id> <amount>",
            "example": "/mortgage borrow prop-a1b2 500",
            "related": "/property /deposit /insure"},
    "insure": {"what": "property insurance: a 30-day policy priced at 5% of the building's current value, in its own currency. District repair waves bill uninsured owners (bank first, then wallet; any shortfall becomes repair debt your rent offsets) — insured buildings are covered in full. Market falls are not insurable. Bare /insure shows your cover and debts.",
            "usage": "/insure · /insure buy <property id> · /insure pay <property id>",
            "example": "/insure buy prop-a1b2",
            "related": "/property /rent /mortgage"},

    "mute": {"what": "mute a member (admins): they can read but not send. Reply to their message with /mute, or name them. A duration like 10m, 2h, 1d sets the length; bare /mute is an hour, 'inf' is indefinite.",
            "usage": "/mute [10m|2h|1d|inf] as a reply, or /mute @member [duration]",
            "example": "/mute @spammer 2h",
            "related": "/unmute /warn /ban"},
    "unmute": {"what": "lift a member's mute (admins).",
            "usage": "/unmute @member, or as a reply",
            "example": "/unmute @ada",
            "related": "/mute"},
    "ban": {"what": "ban a member (admins): out of the group and can't rejoin until unbanned. Reply to their message or name them; anything after the name is the reason, kept in the group log.",
            "usage": "/ban as a reply, or /ban @member [reason]",
            "example": "/ban @spammer flooding",
            "related": "/unban /kick /mute"},
    "unban": {"what": "lift a ban (admins) — the member can rejoin.",
            "usage": "/unban @member",
            "example": "/unban @ada",
            "related": "/ban"},
    "kick": {"what": "kick a member (admins): out of the group now, but free to rejoin. Reply to their message or name them.",
            "usage": "/kick as a reply, or /kick @member",
            "example": "/kick @ada",
            "related": "/ban /warn"},
    "warn": {"what": "warn a member (admins). Warns are counted on the member's card; at the limit (default 3, set in /gsettings) the member is auto-muted for an hour and the count resets.",
            "usage": "/warn as a reply, or /warn @member [reason]",
            "example": "/warn @ada caps",
            "related": "/mute /member"},
    "pin": {"what": "pin a message (admins): reply to the message with /pin.",
            "usage": "/pin as a reply",
            "example": "/pin",
            "related": "/unpin"},
    "unpin": {"what": "unpin the currently pinned message (admins).",
            "usage": "/unpin",
            "example": "/unpin",
            "related": "/pin"},
    "purge": {"what": "delete a message (admins): reply to it with /purge.",
            "usage": "/purge as a reply",
            "example": "/purge",
            "related": "/glog"},
    "report": {"what": "report a member or message to the group admins. The report goes into the held group log with the author's name attached when you reply to their message; admins read it with /glog.",
            "usage": "/report as a reply, or /report @member [reason]",
            "example": "/report @ada spamming",
            "related": "/glog"},
    "cleanup": {"what": "scan this Telegram group for cleanup (admins, claimed groups): deleted accounts (Telegram's own marker) and members with no activity past the group's inactivity window (default 90 days, /gsettings cleanup days N). The scan is a DRY RUN — it removes nothing; it reports who, on what evidence, and what it could not judge. Activity truth is Devon's own member ledger plus Telegram last-seen from her userbot session; the Bot API alone has neither a member list nor last-seen, so on that surface the scan covers the members Devon has records of and says so. Deleted accounts do not wait for this scan: her upkeep cycle removes them automatically in claimed groups where she holds admin power (default on — /gsettings cleanup auto off stops it; the defense autonomy switch governs it like every automatic admin action). Inactive members are removed by command only, never automatically.",
            "usage": "/cleanup [page]",
            "example": "/cleanup",
            "related": "/rmdeleted /rminactive /gsettings /member"},
    "rmdeleted": {"what": "remove the deleted accounts from this Telegram group now (admins, claimed groups): the same removal her upkeep cycle runs automatically — kicks every account Telegram marks deleted, a kick not a ban, journaled per member in /glog with the outcome. Admins, the group owner and Devon's own accounts are never removed; a failure for one account never stops the rest and never reports as success.",
            "usage": "/rmdeleted",
            "example": "/rmdeleted",
            "related": "/cleanup /rminactive /glog"},
    "rminactive": {"what": "remove this Telegram group's inactive members (admins, claimed groups): members with no activity past the inactivity window — the group's setting (/gsettings cleanup days N) or the days given here for this run. Kicked, not banned (they may rejoin), journaled per member in /glog. Members whose activity Devon cannot judge (no record, hidden last-seen) are never removed on a guess; admins, bots and Devon's own accounts are exempt.",
            "usage": "/rminactive [days]",
            "example": "/rminactive 180",
            "related": "/cleanup /rmdeleted /gsettings /member"},
    "tag": {"what": "set a member's tag (§16): a short label — up to 16 characters, no emoji — that Telegram shows beside their name in this group, set by the bot (it needs the manage-tags admin right). Devon records it in the member ledger too, so /member shows it on every surface. A tag is presentation, never power: it changes nothing the member can do. Telegram-only; WhatsApp has no member tags.",
            "usage": "/tag @member <words> | /tag @member off",
            "example": "/tag @ada night owl",
            "related": "/member /boosters"},
    "bansender": {"what": "ban the CHANNEL identity behind a channel post (§16): reply to a message posted as a channel and that channel is banned from this group — while banned, its owner can't post here as any of their channels. This bans the channel, never the person who owns it. Telegram-only.",
            "usage": "reply to a channel post, then /bansender",
            "example": "/bansender",
            "related": "/unbansender /ban"},
    "unbansender": {"what": "lift a sender-chat ban (§16): reply to a post from the banned channel, or name its @handle, and the channel may post here again. Telegram-only.",
            "usage": "/unbansender [@channel]",
            "example": "/unbansender",
            "related": "/bansender /unban"},
    "reactwipe": {"what": "wipe a member's recent reactions in this chat (§16): Telegram clears up to 10,000 of them; the bot needs its delete-messages right. Use it on reaction spam/raids. Telegram-only — WhatsApp gives no way to remove another member's reactions.",
            "usage": "/reactwipe @member (or reply to their message)",
            "example": "/reactwipe @ada",
            "related": "/purge /warn"},
    "boosters": {"what": "this group's boost board (§16): who is boosting the room and how much — Premium boosts, gift-code boosts and giveaways counted as Telegram reports them; a boost whose source names nobody counts for the room and for no member. With a member named, their boosts plus a live per-user lookup on the bot surface. Telegram-only — WhatsApp has no boosts.",
            "usage": "/boosters | /boosters @member",
            "example": "/boosters",
            "related": "/member /tag"},
    "notion": {"what": "Notion: list databases, query rows, create pages — through your connected Notion integration.",
            "usage": "/notion dbs [query] | /notion query <db> [text] | /notion add <page-id> <title> [| <body>]",
            "example": "/notion add <page-id> Shopping list | milk | eggs",
            "related": "/connectors /gcal /trello"},
    "gcal": {"what": "Google Calendar: today's agenda, upcoming days, create events — through your connected Google account.",
            "usage": "/gcal | /gcal agenda [days] | /gcal add <summary> | <YYYY-MM-DD HH:MM> | <end or minutes>",
            "example": "/gcal add Dentist | 2026-10-10 14:00 | 60",
            "related": "/connectors /notion /trello"},
    "trello": {"what": "Trello: boards, lists, cards — create cards, through your connected Trello account.",
            "usage": "/trello boards | /trello lists <board> | /trello cards <list> | /trello add <list> | <name> [| <desc>]",
            "example": "/trello add Todo | buy milk",
            "related": "/connectors /notion /gcal"},
    "exness": {"what": "Exness trading: balance, positions, prices, market orders — through your connected Exness account (orders are your explicit command).",
            "usage": "/exness balance | positions | price <symbol> [tf] | buy <symbol> <lots> [sl] [tp] | sell <symbol> <lots> [sl] [tp] | close <position-id>",
            "example": "/exness buy XAUUSD 0.01",
            "related": "/connectors /stripe /trade"},
    "stripe": {"what": "Stripe payments: balance, customers, charges, hosted checkout links — through your connected Stripe account.",
            "usage": "/stripe balance | customers [email] | charges | link <amount> <currency> <label>",
            "example": "/stripe link 25.50 usd logo gig",
            "related": "/connectors /exness /money"},
    "swarm": {"what": "parallel swarm on one goal — devon builder agents with a fusion of their results, or the research swarm: parallel researchers, trusted sources first, conflicting claims flagged, structured report filed in memory.",
              "usage": "/swarm <goal> [workers]  |  /swarm research <topic>",
              "example": "/swarm audit this repo 4 · /swarm research is termux fast enough for llm inference",
              "related": "/devon /research"},
    "profile": {"what": "the profile-aware runtime: detected environment (termux/mobile/pc/vps/workstation) and every tuned knob — threads, VCPUs, download caps, memory pressure, parallel chats, mission aggressiveness, model preference — with the source of each value.",
                "usage": "/profile", "example": "/profile",
                "related": "/status /workspace"},
    "dns": {"what": "DNS records with zero dependencies: A/AAAA/MX/NS/TXT/SPF/CAA.",
            "usage": "/dns <domain> [record]", "example": "/dns example.com MX",
            "related": "/whois /osint"},
    "scan": {"what": "port scan your own infrastructure — or, handed a URL, the security-audit exposure scan of that web app: exposed files (/.env, /.git), unauthenticated data backends, secrets shipped in its own JS, and open API routes, each finding proven by an evidence pack (the exact request, the full field schema, the source's own record count, a bounded masked sample). Runs as background work; the graded report card + full report file land in DM.",
             "usage": "/scan <target> [ports] [banner] · /scan <url>",
             "example": "/scan https://some-app.example.com",
             "related": "/audit /ports"},
    "audit": {"what": "security audit of a repository (the disclosure desk): secrets in the working tree AND in git history (a deleted key still in history is still leaked), dependency audit against an advisory feed (bundled seed, refreshed from OSV when it answers), and insecure-pattern rules (string-built SQL, eval sinks, wildcard CORS, debug flags, weak password hashes). Every finding is graded by a defined rubric and lands as a report card + full markdown report in DM, with a responsible-disclosure draft when a public contact exists — drafts only; sending is the owner's act. /audit own manages the own-asset registry: targets declared his own are captured unmasked and at full sample depth.",
             "usage": "/audit <repo url|path> · /audit own add|remove|list <asset>",
             "example": "/audit https://github.com/some/project",
             "related": "/scan /owner"},
    "whois": {"what": "domain registration data via RDAP.",
              "usage": "/whois <domain>", "example": "/whois example.com",
              "related": "/dns /osint"},
    "ports": {"what": "what is listening on this machine.",
              "usage": "/ports", "example": "/ports",
              "related": "/scan"},
    "workspace": {"what": "the virtual CPU farm: environment profile, every core's status/load, autoscaling. /workspace [status|scale <n>|up|down|pause|resume <vcpu>|add <kind>|remove <vcpu>].",
              "usage": "/workspace [status|scale <n>|up|down|pause|resume|add|remove]",
              "example": "/workspace",
              "related": "/status /arena"},
    "proxy": {"what": "route outbound traffic through your proxies; free-proxy lab (health-tracked sources + internet-wide discovery, scored + decayed ranking, per-site affinity, protocol-aware picks); SSH→SOCKS5 tunnels.",
              "usage": "/proxy status|list|test|set <url>|clear | scrape|refresh|health|pool [scheme]|pick [udp] [country=XX] | affinity bind|lookup|release|list | discover [seeds] | sources | schedule on [min] | rotate on | ssh start <n> <host> <user>",
              "example": "/proxy health",
              "related": "/scan /osint"},
    "toolhub": {"what": "button tool hub — fresh tested proxies as a file, BIN lookup; per-screen inline buttons with Back navigation (owner-only).",
            "usage": "/toolhub | /toolhub proxy [run] | /toolhub bin <digits> | /toolhub help",
            "example": "/toolhub proxy",
            "related": "/proxy /help"},
    "gen": {"what": "generate a validated automation script of a kind and save it "
                    "to her workspace — the script is syntax-checked before it "
                    "lands. Kinds: backup, cron_sh, dedupe_lines, git_autopush, "
                    "hf_download, jsonl_to_csv, log_rotate, termux_service, "
                    "webhook_notify. Run /gen with no arguments to see every "
                    "kind with its config fields.",
            "usage": "/gen <kind> <name> [json config]  —  /gen (no args) lists the kinds",
            "example": '/gen backup nightly_db {"source": "/data/db", "dest": "/data/backups"}',
            "related": "/code /record"},
    "osint": {"what": "read-only public-intel: target reports, automated campaigns, identity-correlation graph.",
              "usage": "/osint <target> | campaign <seeds…> | graph <verb>…",
              "example": "/osint 8.8.8.8",
              "related": "/dns /whois /proxy"},
    "record": {"what": "capture actions into a reusable macro.",
               "usage": "/record start <name>|stop|step <tool> [json]|status",
               "example": "/record start daily", "related": "/macro /gen"},
    "macro": {"what": "replay a recorded macro (with JSON overrides).",
              "usage": "/macro [list] | <name> [json]", "example": "/macro daily",
              "related": "/record"},
    "file": {"what": "send a file to any live chat.",
             "usage": "/file <platform> <chat> <path> [caption]",
             "example": "/file telegram 123 /sdcard/a.txt",
             "related": "/publish"},
    "publish": {"what": "convert markdown to pdf/html and send it to a chat.",
                "usage": "/publish <platform> <chat> <md> [fmt]",
                "example": "/publish telegram 123 notes.md pdf",
                "related": "/file /deliver"},
    "deliver": {"what": ("create-and-deliver: generate a styled report from a "
                         "topic plus (title, markdown body) sections — HTML "
                         "plus a real PDF — zip them, and send the archive to "
                         "a chat. The send edge is the live gateway's "
                         "file-send path."),
                "usage": ('/deliver report <topic> --section "Title::markdown body" '
                          "[--to platform:chat] [--no-pdf]  (default target: this chat)"),
                "example": '/deliver report "Q3 markets" --section "Overview::The quarter was **volatile**"',
                "related": "/publish /zip"},
    "data": {"what": "training data: mine conversations, list datasets, fetch rows from free HuggingFace sets.",
             "usage": "/data mine [name] | list | fetch <ref> [rows]",
             "example": "/data fetch openai/gsm8k 100",
             "related": "/evolve"},
    "evolve": {"what": "self-improvement, test-gated: propose changes, apply, revert, publish to git.",
               "usage": "/evolve <instruction> | apply <id> | list | audit | revert <id> | git",
               "example": "/evolve make search summaries shorter",
               "related": "/benchmark"},
    "voice": {"what": "the voice catalogue: list voices, switch the active voice live (per chat), speak as it, or clone a new voice from a voice note. Also hosts the unified voice pipeline (denoise → transcribe → edit → multi-speaker TTS → master).",
            "usage": "/voice list | /voice use <name> | /voice say [voice] <text> | /voice clone <name> [path] | /voice transcript <name> <text> | /voice describe <name> <text> | /voice rm <name> | /voice pipeline <audio> [lang] [private|public] | /voice engines | /voice keyterms add <term> [kind] [lang]",
            "example": "/voice pipeline /tmp/voice_note.m4a en private",
            "related": "/speak /stt /audio /clonevoice /voices"},
    "clonevoice": {"what": "clone a voice from a voice note — zero-shot. Attach the audio to the command message.",
            "usage": "/clonevoice <name> (with a voice note / audio attached)",
            "example": "/clonevoice trump",
            "related": "/voices /voice"},
    "voices": {"what": "unified voice picker: every voice across all backends — system (espeak), Piper downloads, and the voice catalogue.",
            "usage": "/voices",
            "example": "/voices",
            "related": "/voice /clonevoice /speak /tts"},
    "speak": {"what": "she says it back to you as a neural voice note.",
              "usage": "/speak <text>", "example": "/speak all done",
              "related": "/tts"},
    "bet": {"what": "the sports bet analyst: ensemble ML (Elo + Poisson + form + market, logistic meta-learner) gives 1X2 probabilities, expected value vs bookmaker odds, and Kelly stakes. Analysis only — it never places bets.",
            "usage": "/bet analyze <home> vs <away> [home_odds draw_odds away_odds] [--league L] | /bet bankroll [set <amount>] | /bet backtest [n] [--seed S] | /bet record <home> <away> <hg-ag>",
            "example": "/bet analyze Arsenal vs Chelsea 2.10 3.40 3.60",
            "related": "/desk"},
    "desk": {"what": "the betting desk: a standing study organ over crash games (Aviator, Stake Crash, bustabit and the likes). It collects timestamped rounds, runs a pattern lab (time-of-day, inter-round timing, autocorrelation, streaks — the time hypothesis, tested), keeps a per-game library of verdicts with evidence, and issues signal calls with at least 2 minutes of lead only off lab-confirmed time structure. Analysis only — it never places bets and never opens book accounts.",
            "usage": "/desk status | /desk study | /desk signals | /desk library | /desk collect | /desk sources | /desk provision",
            "example": "/desk status",
            "related": "/bet"},
    "apk": {"what": "ApkForge, the APK workbench: lands APKs and bundle containers (upload or URL) with full provenance, dissects them statically with androguard + APKiD (manifest anatomy, permissions in plain language, components and what their exported flags expose, dex census, embedded SDKs/trackers, certificate, native libs, packer verdict), teaches from the actual dossier, and diffs two versions component by component (permissions, classes by structural hash, certificate, endpoints). A2 added the editor/builder workbench (journaled edits, fail-closed patches, rebuild + vault-key signing); A3 added version bisect (the exact version where a property flipped, with the transition diff), tracker-signature growth from dissection evidence, native .so hex patching, and the Museum artifact census.",
            "usage": "/apk add <path-or-url> | /apk list | /apk status | /apk show <project> | /apk dissect <project> | /apk teach <project> [topic] | /apk diff <project> <other> | /apk bisect <p1,p2,…> <predicate> | /apk signatures | /apk hexpatch <project> <json> | /apk artifacts | /apk send <artifact> <platform> <chat>",
            "example": "/apk add https://example.com/app.apk",
            "related": "/museum /filesend"},
    "re": {"what": "the reverse-engineering organ: static inspection of binary files under an evidence discipline mined from REA and exceeded — every finding lands as a content-addressed Evidence record (confidence × authority, limitations carried on the record, provenance links to the evidence it rests on), what a parse cannot establish registers as a residual unknown with the authority/confidence closing it would take, and findings journal into the decision journal and memory. Pass 1 is the built-in static spine: ELF (generalizing ApkForge's census parser, plus program headers, needed libraries, static symbols, hardening), PE (imports/exports, certificate table, hardening flags) and Mach-O including fat binaries (load commands, dylibs, symbols, code-signature presence) — parsed by Devon's own parsers, nothing executed. Pass 2 self-provisions the deep rungs (pinned Ghidra headless + JDK, the binutils light rung, the Android reference graph over jadx output); Pass 3 adds firmware (census + pinned unblob extraction) and traffic (pcap/HAR/mitmproxy-flow readers + a probed TShark second voice), all routed by content. Pass 4 is missions + reconstruction verify: a question plus targets becomes a journaled mission — Devon decomposes it over the capability matrix, works the rungs in dependency order, lets the residual unknowns re-drive the plan, and closes with a reconstruction whose every claim is tied to evidence ids and cross-rung verified (contradictions demoted to unknowns, never smoothed). The dynamic rung (Frida/GDB) is not built: questions only running a target could answer open dynamic_required unknowns naming the owner's sandbox-posture gate, and nothing is executed.",
           "usage": "/re status | /re inspect <path> | /re evidence [id] | /re unknowns | /re close <unknown-id> [observed|derived|inferred] <resolution> | /re audit | /re provision | /re graph <apk-or-jadx-dir> [query target] | /re mission <question> -- <target…> | /re missions | /re mstatus <id> | /re mresume <id>",
           "example": "/re inspect /usr/bin/ls",
           "related": "/apk /audit"},
    "pump": {"what": "the pump.fun intelligence desk: a launch firehose (PumpPortal + public Solana RPC + frontend backfill, health-graded on the shared source registry) feeds per-token dossiers built from chain evidence — curve state in SOL (including virtual/real reserve divergence), dev launch/rug history, block-0 bundle absorption, authority flags, holder concentration, wash scores. A computed de-rug gate (INSUFFICIENT_DATA / NO_CALL / CALL_CANDIDATE) decides what is callable; calls publish like signals with entry/target/invalidation/expiry in SOL terms and grade themselves against outcomes in an append-only ledger, hit rate reported against the ~1% graduation base rate. P2 adds a smart-money graph (wallets graded by FIFO realized PnL from observed trades, cohort clustering on co-entry evidence), an attention verifier (organic vs manufactured hype, cross-checked against the real buyer base), a learning loop that recalibrates the score weights from the graded journal in journaled bounded steps, and Library shelving of dossiers/wallets as retrievable knowledge. Signals only — it never buys or sells, holds no wallet keys, signs nothing.",
            "usage": "/pump status | /pump watch | /pump study | /pump dossier <mint> | /pump calls | /pump wallet <addr> | /pump graph | /pump attention <mint> | /pump learn | /pump shelf <query> | /pump sources | /pump watch dev|smart <wallet> | /pump provision",
            "example": "/pump status",
            "related": "/desk /finance"},
    "finance": {"what": "the finance brain: Devon's native TA stack (regime detection, strategy committee, signal fusion, event backtester, risk manager — ported from the Sentinel.py engine) over free keyless market data (Binance/Stooq/Frankfurter). Quotes, regime analysis, signals, full trade plans with stops/targets, per-strategy backtests, and price alerts. Research only — never financial advice.",
            "usage": "/finance quote <sym> [market] | /finance analyze <sym> [market] [tf] | /finance signal <sym> [market] | /finance idea <sym> [market] [--profile P] | /finance backtest <sym> [market] [--profile P] [--strategy NAME] | /finance strategies | /finance compare <s1,s2> [market] | /finance watch <sym> <above|below> <price> [market] | /finance doctor",
            "example": "/finance idea BTC crypto --profile conservative",
            "related": "/money"},
    "task": {"what": "queued instructions she runs and reports back on.",
             "usage": "/task add <instruction> | run [id] | list",
             "example": "/task add check the backups",
             "related": "/schedule /devon"},
    "weather": {"what": "live weather awareness: current conditions, forecast, severe alerts (keyless), plus a USA national situations overview.",
                "usage": "/weather [place] | /weather forecast <place> | /weather alerts [place] | /weather usa | /weather tz",
                "example": "/weather Lagos",
                "related": "/tz /briefing"},
    "tz": {"what": "timezone utilities: your local time labeled, and conversions between zones with DST handling.",
           "usage": "/tz | /tz 2026-10-02 14:00 [from-zone] [to-zone]",
           "example": "/tz 2026-10-02 14:00 America/New_York Africa/Lagos",
           "related": "/weather /briefing"},
    "notify": {"what": "recent alerts, with delivery states.",
               "usage": "/notify [n]",
               "example": "/notify 10", "related": "/proactive"},
    "proactive": {"what": ("she speaks first: the morning briefing and watcher "
                           "alerts are pushed to your DMs. Owner-only, never "
                           "sent to anyone else."),
                  "usage": "/proactive",
                  "example": "/proactive",
                  "related": "/notify /watch"},
    "email": {"what": ("the owner's mailbox over Gmail: list/search/read "
                       "mail, classify unread mail (urgent/action/fyi), "
                       "draft voice-matched replies for review, and "
                       "resurface threads gone quiet 48h. Drafts are never "
                       "auto-sent — /email send <id> shows the exact "
                       "payload and arms it; only your explicit \"send it\" "
                       "sends it."),
              "usage": ("/email triage | inbox [query] [n] | search <query> [n] | "
                        "read <message-id> | thread <thread-id> | drafts | "
                        "compose <to> | <subject> | <body> | "
                        "reply <message-id> | <body> | send <id> | approve <id> | "
                        "discard <id> | followups"),
              "example": "/email inbox from:ada 5",
              "related": "/notify"},
    "gmail": {"what": "alias of /email — the owner's mailbox over Gmail.",
              "usage": "/gmail [same as /email]",
              "example": "/gmail inbox",
              "related": "/email"},
    "mail": {"what": "alias of /email — the owner's mailbox over Gmail.",
             "usage": "/mail [same as /email]",
             "example": "/mail search invoice",
             "related": "/email"},
    "inbox": {"what": ("alias of /email — bare /inbox lists the inbox; "
                       "anything else passes through to /email."),
              "usage": "/inbox [query] [n] | /inbox search <query> | /inbox read <id>",
              "example": "/inbox",
              "related": "/email"},
    "mission": {"what": ("mission progress in chat: % complete, current step, "
                         "an honest ETA, and — when stuck — the concrete "
                         "stall reason. Milestones (started / step / stalled / "
                         "done) also push proactively through the notifier."),
                "usage": ("/mission status [id|name] | /mission results [id|name] | "
                         "/mission list | "
                         "/mission stall <id> <code> <message> | /mission clear <id> | "
                         "/mission pause <id> | /mission resume <id> | "
                         "/mission cancel <id> [reason] | /mission retry <id> | "
                         "/mission watch <id> | /mission unwatch <id> | "
                         "/mission new <research|build|fix> <args>"),
                "example": "/mission status",
                "related": "/devon /notify"},
    "upgrade": {"what": ("the research→approve→evolve review loop in chat: "
                         "research digest findings that clear the ticket "
                         "gate land as upgrade proposals; you review the "
                         "ticket, preview the actual patch, then approve "
                         "(test-gated apply) or deny with a reason. "
                         "Owner-only."),
                "usage": "/upgrade list | /upgrade show <id> | /upgrade diff <id> | "
                         "/upgrade approve <id> | /upgrade deny <id> <reason> | "
                         "/upgrade applied",
                "example": "/upgrade diff upg_9f2k",
                "related": "/evolve /research /notify"},
    "image": {"what": "look up an image (path/URL) or generate one from a text prompt.",
              "usage": "/image <path-or-url> | /image <prompt>",
              "example": "/image a cyberpunk city at night",
              "related": "/lens /look /imggen"},
    "imggen": {"what": "Devon's OWN image studio: native diffusion generation, upscale, checkpoints, training dashboard.",
              "usage": "/imggen <prompt> [--seed N] [--ar 16:9] | /imggen upscale <path> | /imggen checkpoints | /imggen dashboard <run>",
              "example": "/imggen a danfo bus at sunset --ar 16:9",
              "related": "/image"},
    "shorts": {"what": "Devon's short-form content empire: niche → script → voiceover → AI visuals → beat-synced edit → captions → publish.",
              "usage": "/shorts make <niche> \"<topic>\" [--now] | /shorts niches | /shorts status | /shorts calendar | /shorts ledger",
              "example": "/shorts make motivation \"discipline beats motivation\" --now",
              "related": "/imggen /say"},
    "lens": {"what": "reverse image search.",
             "usage": "/lens <path-or-url>", "example": "/lens /sdcard/pic.jpg",
             "related": "/image"},
    "devon": {"what": "the autonomous dev & investigation agent: he plans the tools himself, runs them, digests, replies — for anything you say.",
              "usage": "/devon [free text]",
              "example": "/devon check if the brain replied to the last messages",
              "related": "/swarm /code /task"},
    "think": {"what": "explicit multi-step reasoning with the full trace shown.",
              "usage": "/think <question> [strategy]",
              "example": "/think why is the loop idle auto",
              "related": "/benchmark"},
    "benchmark": {"what": "how sharp the system is right now — runs one quick task "
                         "per dimension and scores 0-1. Dimensions: reasoning, "
                         "planning, tool_use, self_correction (default: all four).",
                  "usage": "/benchmark [reasoning|planning|tool_use|self_correction]",
                  "example": "/benchmark planning",
                  "related": "/evolve"},
    "help": {"what": "this help — the full catalog, or the detail page for one command.",
             "usage": "/help [command]", "example": "/help devon",
             "related": "topic pages: /help budget /help goals /help builds /help skills /help missions /help modes"},
    "list": {"what": "every command you can run from chat, cleanly categorized — this catalog. An optional group name filters it (status, search, building, memory, voice, tools, platform).",
             "usage": "/list [group]", "example": "/list building",
             "related": "/help /commands /menu — nm commands on the console"},
    "commands": {"what": "alias of /list — every executable command, categorized.",
                 "usage": "/commands [group]", "example": "/commands building",
                 "related": "/list /menu"},
    "menu": {"what": "alias of /list — every executable command, categorized.",
             "usage": "/menu [group]", "example": "/menu tools",
             "related": "/list /commands"},
    # wave 72 systems
    "music": {"what": "MusicCreator: composes a REAL song from a topic — "
                      "style-aware lyrics, section structure, chord "
                      "progression, melody description, and a playable .mid "
                      "file (real MIDI, opens in any player).",
              "usage": "/music <topic> [style] [vocals|novocals]  |  /music styles  |  /music song [slug]  |  /music bed <topic> [style]  |  /music full <topic> [style] [voice]  |  /music voices",
              "example": "/music the first rain in lagos lofi",
              "related": "/play (queue the midi or audio) · nm music on the console"},
    "distribute": {"what": "music distribution pipeline: generate → master → "
                           "distribute as one flow. Phase 1 — Devon prepares "
                           "everything (metadata, royalty splits, AI "
                           "disclosure, cover spec, submission checklist); "
                           "the human clicks submit on the distributor. "
                           "Splits are MANDATORY (must sum to 100); AI "
                           "disclosure is always included; never auto-uploads.",
                   "usage": "/distribute [song.wav]  |  /distribute legal",
                   "example": "/distribute ~/music/lagos-nights.wav",
                   "related": "/music full (render the song) · /distribute legal (legal weather)"},
    "play": {"what": "media player: durable queue + transport for audio. "
                     "Uses mpv when installed (full transport, auto-advance, "
                     "survives restarts); otherwise the queue is kept and it "
                     "tells you what to install. In chat, tracks download "
                     "and send as files. Vague queries get a pick-list — "
                     "tap a number (Telegram) or reply with it (WhatsApp).",
              "usage": "/play <song title> | /play <title> by <artist> | /play <url> | /play pick <token> <n> | status | queue | pause | resume | stop | next | prev | seek <s> | volume <n> | remove <n> | clear",
              "example": "/play lifestyle (YA MAN) by ayo maff",
              "related": "/music (make the thing to play) · nm play on the console"},
    "anime": {"what": "anime lookup: the record for a title — format, status, score (AniList, cross-checked against MyAnimeList), episodes, studio, genres, cover art — from the AniList record with Jikan/MyAnimeList as the second seed. Nothing is invented: a title that isn't found says so, naming the sources tried.",
              "usage": "/anime <title> | /anime open <n>",
              "example": "/anime frieren",
              "related": "/manga · /nextep · /recommend · /favorite"},
    "manga": {"what": "manga lookup: the record for a title — status, score, chapters, author, genres, cover art. AniList first, MyAnimeList as the second seed.",
              "usage": "/manga <title> | /manga open <n>",
              "example": "/manga berserk",
              "related": "/anime · /topmanga · /favorite"},
    "anichar": {"what": "anime character lookup (the FEI menu's “character” — Devon's /character is her story cast, so this one is named for anime): who they are, where they're from, their picture. AniList records.",
              "usage": "/anichar <name> | /anichar open <n>",
              "example": "/anichar frieren",
              "related": "/animechars · /topchars · /randomchar"},
    "charpic": {"what": "a random anime character picture, drawn from the public picture pools (waifu.pics / nekos.best seeds) and sent as an image.",
              "usage": "/charpic",
              "example": "/charpic",
              "related": "/waifu · /neko · /kitsune · /husbando"},
    "staff": {"what": "staff lookup: voice actors, directors, writers — their work, the characters they voice, the shows they're known for. AniList records.",
              "usage": "/staff <name> | /staff open <n>",
              "example": "/staff atsumi tanezaki",
              "related": "/anichar · /studio"},
    "studio": {"what": "studio lookup: the studio's record plus its best-known works with scores. AniList records.",
              "usage": "/studio <name> | /studio open <n>",
              "example": "/studio kyoto animation",
              "related": "/anime · /staff"},
    "episodes": {"what": "a title's episodes: the count, the streaming episode list on the record, and the MyAnimeList episode guide (titles + air dates) where filed.",
              "usage": "/episodes <title>",
              "example": "/episodes one piece",
              "related": "/anime · /nextep · /anischedule"},
    "animechars": {"what": "a title's cast: main and supporting characters from the record, with their roles.",
              "usage": "/animechars <title>",
              "example": "/animechars jujutsu kaisen",
              "related": "/anichar · /anime"},
    "trailer": {"what": "a title's trailer from its own record, watchable here through the media organ's /video route.",
              "usage": "/trailer <title>",
              "example": "/trailer chainsaw man",
              "related": "/video · /anime · /streaming"},
    "streaming": {"what": "where to watch a title legally: the streaming links filed on its record (Crunchyroll, Netflix, …). No listed source is stated plainly when the record has none.",
              "usage": "/streaming <title>",
              "example": "/streaming vinland saga",
              "related": "/anime · /trailer"},
    "animereview": {"what": "scores and community reviews for a title (the FEI menu's “review” — Devon's /review is the representation review): AniList score, the cross-checked MAL score, and the top written community reviews, attributed.",
              "usage": "/animereview <title>",
              "example": "/animereview fullmetal alchemist brotherhood",
              "related": "/anime · /recommend"},
    "genres": {"what": "the full genre list on file at AniList — the vocabulary /genre browses.",
              "usage": "/genres",
              "example": "/genres",
              "related": "/genre"},
    "genre": {"what": "the top anime in one genre, ranked by the source's own scores, openable by number.",
              "usage": "/genre <name>",
              "example": "/genre mecha",
              "related": "/genres · /topanime"},
    "topanime": {"what": "the highest-scored anime of all time on AniList, openable by number.",
              "usage": "/topanime",
              "example": "/topanime",
              "related": "/topmanga · /popular · /trending"},
    "topmanga": {"what": "the highest-scored manga of all time on AniList, openable by number.",
              "usage": "/topmanga",
              "example": "/topmanga",
              "related": "/topanime · /manga"},
    "popular": {"what": "the most popular anime right now (or manga with /popular manga), by AniList's popularity field.",
              "usage": "/popular [manga]",
              "example": "/popular",
              "related": "/trending · /topanime"},
    "trending": {"what": "what's trending right now on AniList (or manga with /trending manga).",
              "usage": "/trending [manga]",
              "example": "/trending",
              "related": "/popular · /airing"},
    "airing": {"what": "anime airing this season, biggest first, each with its next-episode countdown where the schedule is published.",
              "usage": "/airing",
              "example": "/airing",
              "related": "/anischedule · /nextep · /upcoming"},
    "upcoming": {"what": "announced anime that isn't out yet, most anticipated first.",
              "usage": "/upcoming",
              "example": "/upcoming",
              "related": "/airing · /season"},
    "season": {"what": "one season's anime (winter/spring/summer/fall + year), biggest first. Bare /season shows the current season.",
              "usage": "/season [season] [year]",
              "example": "/season fall 2025",
              "related": "/airing · /upcoming"},
    "topchars": {"what": "the most-loved anime characters on AniList, ranked by favourites, openable by number.",
              "usage": "/topchars",
              "example": "/topchars",
              "related": "/anichar · /randomchar"},
    "anischedule": {"what": "the week's airing grid (the FEI menu's “schedule” — Devon's /schedule is her cron surface): every episode airing in the next 7 days, grouped by day, times read in WAT (UTC+1).",
              "usage": "/anischedule",
              "example": "/anischedule",
              "related": "/airing · /nextep"},
    "nextep": {"what": "when a title's next episode airs: the date and time in WAT plus a live countdown, computed from the record's airing schedule. A finished show says so; an unpublished schedule says so.",
              "usage": "/nextep <title>",
              "example": "/nextep one piece",
              "related": "/anischedule · /airing · /episodes"},
    "recommend": {"what": "more like a title you name: the recommendations filed on its AniList record (or, when none are filed, the top of its first genre — labelled as such). Real records only, openable by number.",
              "usage": "/recommend <title>",
              "example": "/recommend cowboy bebop",
              "related": "/anime · /animereview"},
    "randomanime": {"what": "a random anime record from the whole AniList catalogue — the full card, cover included.",
              "usage": "/randomanime",
              "example": "/randomanime",
              "related": "/randommanga · /randomchar"},
    "randommanga": {"what": "a random manga record from the whole AniList catalogue.",
              "usage": "/randommanga",
              "example": "/randommanga",
              "related": "/randomanime · /manga"},
    "randomchar": {"what": "a random character record from AniList — who they are and where they're from.",
              "usage": "/randomchar",
              "example": "/randomchar",
              "related": "/anichar · /topchars"},
    "animequote": {"what": "an anime quote with its attribution (character + show), via animechan. Name a title to draw from that show's filed quotes.",
              "usage": "/animequote [title]",
              "example": "/animequote naruto",
              "related": "/anime"},
    "favorite": {"what": "YOUR anime favorites — Devon's own per-person list in her store (not an AniList/MAL account action). Add resolves the title against the real records first; only real records can be saved. Only you ever see your list.",
              "usage": "/favorite | /favorite add <title> | /favorite remove <n|title> | /favorite open <n>",
              "example": "/favorite add frieren",
              "related": "/anime · /manga"},
    "whatanime": {"what": "reverse identification: reply to an anime screenshot/frame (or paste an image URL) and get the anime, the episode and the timestamp in it, via trace.moe. Matches under the confidence floor are reported as no-match, never dressed up as answers.",
              "usage": "/whatanime (as a reply to a picture) | /whatanime <image url>",
              "example": "/whatanime",
              "related": "/anime · /lens"},
    "waifu": {"what": "a waifu picture from the public pools (waifu.pics, nekos.best), sent as an image.",
              "usage": "/waifu",
              "example": "/waifu",
              "related": "/charpic · /neko · /husbando"},
    "neko": {"what": "a neko picture from the public pools, sent as an image.",
              "usage": "/neko",
              "example": "/neko",
              "related": "/charpic · /waifu · /kitsune"},
    "kitsune": {"what": "a kitsune picture from the public pools, sent as an image.",
              "usage": "/kitsune",
              "example": "/kitsune",
              "related": "/neko · /charpic"},
    "husbando": {"what": "a husbando picture from the public pools, sent as an image.",
              "usage": "/husbando",
              "example": "/husbando",
              "related": "/waifu · /charpic"},
    "download": {"what": "any-URL media downloader: paste a link, get the file. "
                         "YouTube, TikTok, Instagram, X/Twitter, SoundCloud, "
                         "direct mp3/mp4 links — yt-dlp handles 1000+ sites, "
                         "with proxy and browser fallback stages. Spotify links "
                         "identify-and-match (full track from a real source); "
                         "TikTok/Instagram links take their dedicated routes. "
                         "The file is sent to this chat.",
                 "usage": "/download <url> [audio]",
                 "example": "/download https://youtube.com/watch?v=... audio",
                 "related": "/play (play/queue audio) · /video download (video search download) · /tiktok · /instagram · /spotify · /aio"},
    "tiktok": {"what": "TikTok downloader: paste a TikTok link (share links "
                       "included), get the video file. Fetches the "
                       "no-watermark rendition when a route genuinely "
                       "provides one and says so; when the watermarked "
                       "rendition is all a route has, the reply says that "
                       "instead — the label is never upgraded.",
               "usage": "/tiktok <url> [audio]",
               "example": "/tiktok https://vm.tiktok.com/ZMhAbCdEf/",
               "related": "/download (any link) · /aio (any supported link)"},
    "ttdl": {"what": "alias of /tiktok — TikTok downloader.",
             "usage": "/ttdl <url> [audio]",
             "example": "/ttdl https://www.tiktok.com/@user/video/1234567890",
             "related": "/tiktok · /download"},
    "instagram": {"what": "Instagram downloader: public reels and posts "
                          "come down as files. Private, story and "
                          "age-gated posts hit Instagram's login wall — the "
                          "reply names the wall; Devon holds no Instagram "
                          "login for downloads and invents none.",
                  "usage": "/instagram <url>",
                  "example": "/instagram https://www.instagram.com/reel/AbCdEfGh/",
                  "related": "/download (any link) · /aio (any supported link)"},
    "spotify": {"what": "Spotify identify-and-match: a Spotify link names "
                        "the track/album/playlist (from Spotify's public "
                        "pages — Spotify's own audio is DRM 30s previews, "
                        "never sent), then the FULL track is fetched "
                        "through the /play source ladder and labeled by "
                        "its real source ('matched via YouTube'). "
                        "Playlists/albums resolve track by track (cap 10) "
                        "with a matched/failed summary.",
                "usage": "/spotify <url>",
                "example": "/spotify https://open.spotify.com/track/0TnOYISbd1XYRBk9myaseg",
                "related": "/play (the ladder it matches through) · /download (routes Spotify links here)"},
    "aio": {"what": "all-in-one downloader: paste any supported link — "
                    "TikTok, Instagram, Spotify, YouTube, SoundCloud, "
                    "X/Twitter, Facebook, Vimeo, Dailymotion, Twitch, "
                    "Reddit, Audiomack, Bandcamp, or a direct media file — "
                    "and it routes to the right fetch, the same routes "
                    "/download uses. A site no route reaches gets an "
                    "honest refusal up front.",
            "usage": "/aio <url> [audio]",
            "example": "/aio https://www.tiktok.com/@user/video/1234567890",
            "related": "/download (the universal surface) · /tiktok · /instagram · /spotify"},
    "video": {"what": "VideoFinder: finds videos across the open web — "
                      "multi-engine search, ranked by video-URL confidence + "
                      "relevance, enriched with oEmbed (author/thumbnail) and "
                      "yt-dlp metadata (duration). Downloads real files.",
              "usage": "/video <query> [platform] | /video download <url> [audio] | /video platforms",
              "example": "/video lofi beats for studying youtube",
              "related": "/searchdeep · nm video on the console"},
    "director": {"what": "Devon directs: professional script coverage, "
                      "scene/shot breakdown, lighting + camera design "
                      "per shot (primitives composed by style packs — "
                      "the engine bakes no look), generation through "
                      "the videogen registry, and a frame/Seer QA loop "
                      "with one flagged regeneration per failed shot.",
              "usage": "/director cover <script> | plan <script> [--style PACK] | shoot <production-id> | regen <prod-id> <shot-id> | show [prod-id]",
              "example": "/director plan the-letter.txt --style noir",
              "related": "/video gen (single clips) · nm director on the console"},
    "caption": {"what": "Burn AI subtitles into a video: faster-whisper transcription → styled captions → FFmpeg burn-in.",
              "usage": "/caption [style] — attach a video or pass a path",
              "example": "/caption karaoke",
              "related": "/video"},
    "vision": {"what": "Devon's eyes: describe, read-text, locate, compare, analyze (native), info.",
              "usage": "/vision [subcommand] <image> [args] — attach an image or pass a path/URL "
                       "(describe, read-text, locate, compare, analyze, faces, qr, layout, "
                       "hash, exif, colors, info)",
              "example": "/vision what's in this screenshot?",
              "related": "/look /lens"},
    "exec": {"what": "Execution system: runs code in the sandbox with "
                     "structured results (stdout, stderr, exit code, wall "
                     "time, timeout flag, files written). Languages: python, "
                     "javascript, bash, ruby, perl, php, lua, tcl, awk, "
                     "julia, deno, bun, c, cpp, rust, go — whichever is "
                     "installed. Network off by default.",
              "usage": "/exec <code> [lang] | /exec languages",
              "example": "/exec print(sum(range(10))) python",
              "related": "/code (draft→run→fix agent) · nm exec on the console"},
    "zip": {"what": "Archive system: identifies archives by magic bytes "
                    "(a renamed file still works), lists/creates/extracts "
                    "zip, tar, tar.gz, tar.bz2, tar.xz, gz, bz2, xz (7z/rar "
                    "when the tools are installed). Extraction is "
                    "traversal-safe — .. escapes are skipped and reported.",
              "usage": "/zip <paths…> --dest x.zip | /zip list|info|extract <archive> | /zip digest <archive> | /zip compress <file> [fmt]",
              "example": "/zip workspace/a.txt workspace/b.txt --dest bundle.zip",
              "related": "nm zip on the console · /zip digest curates the docs into memory"},
    "apps": {"what": "Builder system: scaffolds complete RUNNABLE apps "
                     "(static site, Flask, FastAPI, Express, React+Vite, "
                     "Python CLI) in workspace/apps/ — real routes, state, "
                     "CSS, manifest, and post-build validation "
                     "(py_compile / node --check / JSON).",
              "usage": "/apps build <name> --stack <stack> | /apps serve <name> [--port N] | /apps stop <name> | /apps served | /apps [list|stacks|info <name>]",
              "example": "/apps build mytodo --stack fastapi  then  /apps serve mytodo",
              "related": "/code · nm apps on the console"},
    "hub": {"what": "MediaHub: the one-call media orchestrator. song = compose a "
                    "real song then queue+play its MIDI; video = find → download → "
                    "queue → play; podcast = find → download audio → transcribe (STT "
                    "when a backend is configured) → extractive/model summary → "
                    "titled chapters, transcript saved under podcasts/.",
            "usage": "/hub song <topic…> [style] | /hub video <query…> [platform] | /hub podcast <query…> | /hub status",
            "example": "/hub song the first rain in lagos lofi",
            "related": "/podcast (shortcut) · /music · /play · /video · nm hub on the console"},
    "podcast": {"what": "The podcast pipeline, one call: find a video/podcast for "
                        "the query, download the audio, transcribe it (OpenAI-"
                        "compatible STT or whisper.cpp when available), summarize, "
                        "split into titled chapters, and save the transcript. "
                        "Sends the transcript file back to this chat when a send "
                        "target is configured.",
            "usage": "/podcast <query…> [platform]",
            "example": "/podcast developer keynote 2026",
            "related": "/hub podcast (same pipeline) · /stt · nm hub on the console"},
    "dj": {"what": "Radio DJ: builds a Devon FM show from today's trending charts "
                   "plus your own generated tracks — DJ voice breaks, chart "
                   "shoutouts, crossfades and filter-sweep transitions, one WAV.",
            "usage": "/dj [trending|<genre>]",
            "example": "/dj uk-drill",
            "related": "/music · /play · /podcast"},
    "produce": {"what": "Producer: analyzes a Spotify track's musical DNA "
                        "(tempo, key, energy, mood) and composes an ORIGINAL "
                        "piece in its lane — motivic writing, classical "
                        "development techniques, composed transitions. "
                        "Never copies the reference's melody.",
                "usage": "/produce <spotify-url> | /produce <vibe words…>",
                "example": "/produce dark driving edm like black out days",
                "related": "/music · /dj · /play"},
    "like": {"what": "Rate the last produced track positively — Devon learns your taste.",
             "usage": "/like [notes…]",
             "example": "/like the drop goes hard",
             "related": "/dislike · /produce"},
    "dislike": {"what": "Rate the last produced track negatively — Devon learns what to avoid.",
                "usage": "/dislike [notes…]",
                "example": "/dislike too slow",
                "related": "/like · /produce"},
    "cookies": {"what": "YouTube cookie file status — needed when YouTube blocks downloads.",
                "usage": "/cookies [check]",
                "example": "/cookies",
                "related": "/play"},
    "sham": {"what": "Identify the song in a voice note or audio — reply to it with /sham, or send audio with /sham as the caption.",
                "usage": "/sham",
                "example": "/sham (reply to a voice note)",
                "related": "/play"},
    "musicnote": {"what": "Answer the music voice-note ask — Find this music (identify the song) or Create a song (produce a new song from it). Owner-only; the ask's buttons call this.",
                "usage": "/musicnote find | create",
                "example": "/musicnote find",
                "related": "/sham · /music"},
    "fix": {"what": "CI loop: runs the code in the sandbox, and while it fails the "
                    "model rewrites it and it runs again, until exit 0 (or the "
                    "expected text appears in stdout). Reports every round — exit "
                    "code, timing, what the model did. Without an LLM backend it "
                    "honestly reports the last failure instead of pretending.",
            "usage": "/fix <code> [lang] [--rounds N]",
            "example": "/fix print(total) python",
            "related": "/exec (single run) · nm exec until_green on the console"},
}

#: The catalog's groups — every registered command must appear in one.
_HELP_GROUPS: list[tuple[str, list[str]]] = [
    ("her day", ["status", "ping", "platforms", "proposals", "approve", "deny",
                 "mood", "stage", "model", "providers", "say", "power", "quit",
                 "mode", "agent", "mind", "pulse"]),
    ("search & research", ["search", "searchdeep", "searchleads",
                           "money",
                           "searchhist", "research", "news", "osint",
                           "dns", "scan", "audit", "whois", "ports",
                           "brief", "competitor", "aeo"]),
    ("wisdom keeper — corpus · history · practice", ["wisdom", "wis"]),
    ("building for real", ["code", "py", "devon", "swarm", "task", "gen",
                           "data", "evolve", "upgrade", "arena", "trial",
                           "identity", "book", "books", "features",
                           "checkpoint", "checkpoints", "rewind", "novel"]),
    ("memory & thinking", ["remember", "recall", "forget", "memories", "think",
                           "benchmark", "redteam", "review", "tutor", "learn", "ekiti", "course", "health", "train", "form", "film", "graph", "route", "eta", "congestion", "immune", "impossible",
                           "people", "commitments", "decisions", "iddocs", "focus"]),
    ("money & spending", ["spend", "budget", "spending", "mandate",
                          "balances", "goal", "insights", "alert"]),
    ("games — 41, DM + group, start them directly",
     ["game", "inventory", "equip", "unequip", "repair", "level",
      "skill", "title", "stats", "daily", "mastery", "gift",
      "wordchain", "hangman", "numberguess", "two_truths", "wyrr",
      "spy", "auction", "trivia", "mafia", "king", "story", "rpg", "shop",
      "duel", "case", "world", "escape", "political",
      "poker", "ttt", "bulls", "craps", "memory", "mines", "wordle",
      "2048", "snake", "connect4", "battleship",
      "blackjack", "roulette", "slots", "miniapp", "gtrip", "predict",
      "gomoku", "reversi", "checkers", "20q", "rps", "digits",
      "sudoku", "anagram", "cryptogram", "math", "npc", "dm", "charagent",
      "pvp", "raid"]),
    ("voice & vision", ["tts", "speak", "stt", "voice", "look", "image", "lens",
                        "vision", "voices", "clonevoice", "sing",
                        "translate", "tl"]),
    ("decoding & crypto", ["decode", "cookies", "cookielab", "structure", "cipher",
                           "monitor", "investigate"]),
    ("media system", ["music", "distribute", "play", "video", "director", "hub", "podcast", "dj", "produce", "like", "dislike", "cookies",
                      "download", "tiktok", "ttdl", "instagram", "spotify", "aio", "shorts", "caption", "imggen", "sham", "musicnote",
                      "steal", "sticker", "convert", "gif", "emoji", "customemoji", "emojiart", "react"]),
    ("execution · archives · builders", ["exec", "zip", "apps", "fix", "deliver"]),
    ("tools & automation", ["schedule", "db", "artifact", "api", "connectors", "provision", "pair", "owner", "sudo", "massreact", "pollvote", "chfollow", "notion",
                            "gcal", "trello", "proxy", "workspace", "record",
                            "macro", "file", "publish", "notify", "proactive", "mission",
                            "bet", "desk", "apk", "re", "pump", "finance", "weather", "tz", "email", "gmail", "mail", "inbox", "routine", "home", "store", "track", "untrack", "trip", "travelclient", "contract", "legal", "research", "contracts", "regwatch", "uprofile", "vnote", "match", "cgroup", "challenge", "room", "scamcheck", "passport", "truecost", "value", "valuepick", "watch", "send", "leads", "guardrails", "competitor", "cost", "panel",
                           "exness", "stripe", "content",
                           "tempmail", "checkmail", "vnumber", "checksms",
                           "browser", "paper", "backup"]),
    ("platform control", ["start", "stop", "profile", "chatmode", "style",
                          "doctor",
                          "update", "pull", "deploy", "restart"]),
    ("groups & communities", ["rules", "member", "glog", "gsettings",
                            "strikes", "recheck", "joinq", "console",
                            "undonotice", "flag", "rep", "repclear",
                            "repdismiss",
                            "balance", "mine", "work", "pay", "gshop",
                            "quests", "treasury", "drop", "claim",
                            "property", "buyproperty", "sellproperty",
                            "rent", "listproperty", "auctions",
                            "mortgage", "insure",
                            "gear", "heist", "shoot", "kill",
                            "assassinate", "sabotage", "gang", "war",
                            "police", "bounty", "claimdrop",
                            "poll", "vote", "milestones",
                            "contributions", "boards", "webshot",
                            "mute", "unmute", "ban", "unban", "kick",
                            "warn", "pin", "unpin", "purge", "report",
                            "cleanup", "rmdeleted", "rminactive",
                            "tag", "bansender", "unbansender",
                            "reactwipe", "boosters", "party", "ajo"]),
    ("discovery", ["list", "commands", "menu", "help", "tour", "heal"]),
]

#: Stacking topic page (shared by /help stack and /help stacking).
_STACK_TOPIC_PAGE = (
    "stacking — several commands in one message\n"
    "  one command per line; every line must be a full /command:\n"
    "  /status  (line 1)\n"
    "  /mood    (line 2)\n"
    "  runs in order; each step's reply is shown under a\n"
    "  '── 1/2 /status ──' header.\n"
    "  * {prev} in a later step is replaced by the previous step's\n"
    "    output (truncated); {prev:1} = step 1's output; {{prev}}\n"
    "    stays literal.\n"
    "  * a step that fails, is denied, or asks for confirmation\n"
    "    STOPS the stack — later steps do not run, and the stop is\n"
    "    reported, never silent.\n"
    "  * gating is checked for EVERY step before anything runs:\n"
    "    an outsider stacking an owner-only command gets an explicit\n"
    "    denial and nothing runs.\n"
    "  * max 5 commands per message. Games, and quit / forget /\n"
    "    restart / deploy / update / pull, can never be stacked.\n"
    "  * ';' and '&&' do NOT stack — they belong to /route,\n"
    "    /overview, /exec, /py, URLs and quoted args.\n"
    "  * inside code fences or \"quoted strings\", a line is argument\n"
    "    text, never a new command.")


_TOPIC_PAGES: dict[str, str] = {
    "stack": _STACK_TOPIC_PAGE,
    "stacking": _STACK_TOPIC_PAGE,
    "budget": (
        "model budget — how she spends your model calls\n"
        "  default: UNLIMITED (0).  If you set a daily cap, everything "
        "plans around it:\n"
        "  * builds RESERVE their calls before drafting — if today can't "
        "afford them the project SUSPENDS (honest report, nothing burned)\n"
        "  * the next heartbeat RESUMES it automatically after the day "
        "rolls over\n"
        "  * 'nm autonomy budget' shows used / reserved / remaining\n"
        "  * remove the cap:  nm autonomy budget --unlimited\n"
        "  * set a cap:       nm autonomy budget --cap 100\n"
        "  * the loop self-throttles its heartbeat near the limit\n"
        "nothing is throttled unless you put a cap there."),
    "goals": (
        "goals & projects — long-term work that survives restarts\n"
        "  /think and /devon start GOALS; goals get PROJECTS; projects "
        "EXECUTE.\n"
        "  * a goal is a multi-step plan with priority + dependencies\n"
        "  * the heartbeat advances the highest-value goal each tick\n"
        "  * CLI:  nm goal list|create|advance|status   ·   nm project "
        "list|run|report\n"
        "  * she self-heals failed projects (rewrites the step with the "
        "lessons she learned), up to a cap, then hands it to you\n"
        "  * finished goals distill their lessons into memory (skills + "
        "knowledge graph)"),
    "builds": (
        "building for real — /code, /devon builds, project steps\n"
        "  she drafts code, RUNS it in the sandbox, reads the real error, "
        "fixes — until the ACCEPTANCE command exits 0.\n"
        "  * acceptance is a real shell command (e.g. 'python3 app.py "
        "--check'), compiled from what you asked for\n"
        "  * every passing acceptance is FROZEN; any later step re-runs "
        "all earlier acceptances — regressions are caught (nm project "
        "regress <id>)\n"
        "  * self-corrected sessions become SKILLS named after the error "
        "— the next similar build recalls the proven fix\n"
        "  * files land in her workspace; /file sends them to you"),
    "skills": (
        "skills — her memory of how to do things\n"
        "  every success and every fixed failure is captured: tool "
        "sequences, working strategies, code fixes, lessons.\n"
        "  * she recalls relevant skills BEFORE starting (prior art in "
        "the prompt)\n"
        "  * repeat errors become SYSTEMIC TRAPS — new builds are warned "
        "up front\n"
        "  * usage is tracked: what worked gets recalled more\n"
        "  * CLI:  nm skill list|save|recall|use|stats\n"
        "  * /recall <query> — ask her what she remembers"),
    "missions": (
        "missions — the whole portfolio, cost-aware\n"
        "  the heartbeat picks work by EXPECTED VALUE: priority tempered "
        "by dependency depth, heal history, remaining size and MODEL "
        "COST.\n"
        "  * a 3-step build (9 model calls) ranks below a 3-step "
        "research (3 calls) at the same priority\n"
        "  * with a budget cap: 'nm mission plan' shows per goal what "
        "fits today (steps affordable) and the ETA in days\n"
        "  * 'nm mission next' — what she will do at the next heartbeat\n"
        "  * /mission status [id|name] — chat-visible: % complete, current "
        "step, ETA, and the concrete stall reason when stuck "
        "(milestones also push proactively: started / step / stalled / done)"),
    "modes": (
        "modes — how autonomous she is\n"
        "  /mode off      asleep — only answers when you message\n"
        "  /mode suggest  works, but sensitive actions wait for /approve\n"
        "  /mode auto     acts on her own within the rules\n"
        "  /power on <key>  power mode: no human-like pacing, maximum "
        "capability (key from your config; or /power on <identity> "
        "<passphrase> with the owner seal)\n"
        "  /stage …     relationship stage tunes tone & initiative\n"
        "  normal mode keeps protective pacing (DMs/groups/hourly); "
        "power mode removes it — that's the only switch."),
}


#: Direct game-start commands (wave 87) get real detail pages too — one
#: per game kind that lives in CONTROL_COMMANDS.
_GAME_HELP: dict[str, str] = {
    "poker": "heads-up limit poker vs the table (Devon deals).",
    "ttt": "tic-tac-toe, 3×3, best of five.",
    "bulls": "bulls & cows — guess the 4-digit number, clues per try.",
    "craps": "craps table — pass/don't\'t pass, come-out rolls and points.",
    "memory": "memory match — flip pairs, fewest moves wins.",
    "mines": "minesweeper cashout — step on a mine and the pot is gone.",
    "wordle": "wordle — five letters, six guesses, shared daily word.",
    "2048": "2048 — slide and merge to 2048 on a 4×4 board.",
    "snake": "snake — eat, grow, don\'t bite yourself.",
    "connect4": "connect four — drop discs, line four up.",
    "battleship": "battleship — your fleet vs hers, 10×10.",
    "blackjack": "blackjack vs the dealer, insurance off, double allowed. /blackjack <amount> deals a hand at that stake.",
    "roulette": "roulette — red/black/odd/even/straight-up bets.",
    "slots": "three-reel slots — spin for combos, jackpot pays big. /slots <amount> places that bet and spins in one command.",
    "gomoku": "gomoku — five in a row on 15×15, async, no clock.",
    "reversi": "reversi (othello) — outflank and flip, most discs wins.",
    "checkers": "english draughts — forced jumps, kings, async.",
    "sudoku": "sudoku — 9×9, real generated puzzles, 3 strikes and you're out.",
    "anagram": "anagram — 6-round unscramble race, fastest fingers win.",
    "cryptogram": "cryptogram — crack the substitution cipher, 3-quote race.",
    "20q": "20 questions — she thinks of something, you ask yes/no (20 max).",
    "rps": "rock paper scissors — best of five vs the house.",
    "digits": "digit memory — repeat the digits back, they grow every round.",
}
def _wrap_commands(kinds: list[str], width: int = 64) -> list[str]:
    """Wrap ``/cmd`` tokens so a catalog line never runs long in chat.

    The games group alone lists 47 entries — one line of that is a wall
    of text on a phone screen, so wrap at ~64 columns.
    """
    lines: list[str] = []
    cur = ""
    for kind in kinds:
        token = "/" + kind
        if cur and len(cur) + len(token) + 2 > width:
            lines.append("  " + cur)
            cur = ""
        cur = f"{cur}  {token}" if cur else token
    if cur:
        lines.append("  " + cur)
    return lines


def _detailed_overview(*, public_only: bool = False) -> str:
    if public_only:
        # The gated public surface's catalog: the PUBLIC command list
        # only. Owner-only commands (platform control, money, trial
        # tooling, …) are not advertised there — groups left empty by
        # the filter drop out entirely, and the owner-system tail
        # (topic pages, console) is not public copy.
        lines = [
            "the public catalog — detail for any command: /help <command>",
            "",
        ]
        groups = [(group, [k for k in kinds if k in PUBLIC_COMMANDS])
                  for group, kinds in _HELP_GROUPS]
        groups = [(group, kinds) for group, kinds in groups if kinds]
    else:
        lines = [
            "the full catalog — detail for any command: /help <command>",
            "topic pages: /help budget|goals|builds|skills|missions|modes|stack",
            "",
        ]
        groups = _HELP_GROUPS
    for group, kinds in groups:
        lines.append(f"  — {group} —")
        lines.extend(_wrap_commands(kinds))
        lines.append("")
    shown = {k for _, kinds in groups for k in kinds}
    rest = [k for k in CONTROL_COMMANDS if k not in shown]
    if public_only:
        rest = [k for k in rest if k in PUBLIC_COMMANDS]
    if rest:
        lines.append("  — everything else —")
        lines.append("  " + "  ".join(f"/{k}" for k in rest))
        lines.append("")
    if not public_only:
        lines += [
            "how she works:  goals → projects → real execution.  Builds run "
            "in a sandbox until the real acceptance command passes; every "
            "fixed failure becomes a skill; the portfolio is ranked by "
            "cost-aware expected value; the budget is unlimited unless you "
            "cap it (nm autonomy budget).",
            "console:  the whole system is also `nm <command>` — "
            "`nm help <topic>` shows these same pages.",
        ]
    return "\n".join(lines)


def _public_no_help_page(t: str) -> str:
    """The /help answer for a non-public topic on the gated surface."""
    return (f"no help page for '{t}' here — this bot carries the public "
            f"commands only. /help shows what I can do here.")


#: One-line summaries for /list — what each command actually DOES.
#: (COMMAND_DETAILS already carries richer pages; /list stays scannable.)
LIST_GROUPS: list[tuple[str, list[str]]] = [
    ("her day — state & control",
     ["status", "ping", "platforms", "mood", "stage", "model", "providers", "say", "power",
      "proposals", "approve", "deny", "start", "stop", "quit",
      "mode", "agent", "chatmode", "style", "doctor", "profile", "mind",
      "spend", "budget", "spending", "mandate", "balances", "goal",
      "insights", "alert", "cost", "heal", "tour", "pulse"]),
    ("search & research",
     ["search", "searchdeep", "searchleads", "money", "searchhist", "research",
      "news", "osint", "dns", "scan", "audit", "whois", "ports",
      "brief", "competitor", "aeo"]),
    ("wisdom keeper — corpus · history · practice",
     ["wisdom", "wis"]),
    ("building for real — code & missions",
     ["code", "py", "devon", "swarm", "task", "gen", "data", "evolve",
      "upgrade", "arena", "trial", "identity", "book", "exec", "apps", "fix", "structure",
      "deliver", "checkpoint", "checkpoints", "rewind", "novel", "books"]),
    ("media system — music · playback · video · podcast",
     ["music", "play", "video", "director", "hub", "podcast", "zip", "audio", "overview", "character", "audiobook",
      "distribute", "dj", "download", "tiktok", "ttdl", "instagram", "spotify", "aio", "produce", "like", "dislike",
      "shorts", "caption", "imggen", "sham", "musicnote",
      "steal", "sticker", "convert", "gif", "emoji", "customemoji",
      "emojiart", "react",
      "awoo", "baka", "bite", "blush", "bonk", "bored",
      "bully", "cringe", "cry", "cuddle", "dance", "facepalm",
      "feed", "glomp", "handhold", "handshake", "happy", "highfive",
      "hug", "kiss", "laugh", "lick", "nom", "pat",
      "poke", "pout", "shrug", "slap", "sleep", "smile",
      "smug", "stare", "thumbsup", "tickle", "wave", "wink",
      "yeet"]),
    ("anime — lookup · schedules · pictures",
     ["anime", "manga", "anichar", "charpic", "staff", "studio",
      "episodes", "animechars", "trailer", "streaming", "animereview",
      "genres", "genre", "topanime", "topmanga", "popular", "trending",
      "airing", "upcoming", "season", "topchars", "anischedule",
      "nextep", "recommend", "randomanime", "randommanga", "randomchar",
      "animequote", "favorite", "whatanime",
      "waifu", "neko", "kitsune", "husbando"]),
    ("fun one-shots — jokes · facts · pictures · space",
     ["joke", "dadjoke", "progjoke", "chuck", "fact", "fact2",
      "catfact", "numfact", "advice", "quote",
      "dog", "cat", "fox", "meme", "riddle", "8ball", "choose",
      "activity", "pokemon", "history", "nasa", "calc"]),
    ("makers & tools — cards · qr · links · bible",
     ["qcard", "canvasquote", "iqc", "wndwqute", "meme2",
      "fakemsg", "fakettc", "nokiamsg", "fakeff", "fakeffduo",
      "ffgirl", "removebg", "qr", "shorturl", "unshorten",
      "bible", "couplepp", "pp"]),
    ("chat & privacy — your chats · your accounts",
     ["archive", "unarchive", "mutechat", "unmutechat",
      "readchat", "unreadchat", "star", "unstar",
      "forward",
      "unsend", "edit", "editmsg", "antiedit",
      "block", "unblock", "anticall", "disappear",
      "statusreact", "poststatus", "chanstory", "banchecker", "gcstatus", "statusinfo",
      "pollres", "tagall", "hidetag", "antitag", "viewonce", "vv",
      "joinlink", "newgroup", "newchannel", "leave"]),
    ("memory & thinking",
     ["remember", "recall", "forget", "memories", "think", "benchmark", "redteam",
      "review", "tutor", "learn", "ekiti", "course", "health", "train", "form", "film", "graph", "route", "eta", "congestion",
      "immune", "impossible", "people", "commitments", "decisions", "iddocs", "focus"]),
    ("games — 41, DM + group, start them directly",
     ["game", "inventory", "equip", "unequip", "repair", "level",
      "skill", "title", "stats", "daily", "mastery", "gift",
      "wordchain", "hangman", "numberguess", "two_truths", "wyrr",
      "spy", "auction", "trivia", "mafia", "king", "story", "rpg", "shop",
      "duel", "case", "world", "escape", "political",
      "poker", "ttt", "bulls", "craps", "memory", "mines", "wordle",
      "2048", "snake", "connect4", "battleship",
      "blackjack", "roulette", "slots", "miniapp", "gtrip", "predict",
      "gomoku", "reversi", "checkers", "20q", "rps", "digits",
      "sudoku", "anagram", "cryptogram", "math", "npc", "dm", "charagent",
      "pvp", "raid"]),
    ("voice & vision",
     ["tts", "speak", "stt", "voice", "look", "image", "lens",
      "vision", "voices", "clonevoice", "sing", "translate", "tl"]),
    ("tools & automation",
     ["schedule", "db", "artifact", "api", "proxy", "toolhub", "workspace", "record", "macro", "file",
      "publish", "notify", "proactive", "mission", "features", "decode", "cookies", "cookielab", "cipher",
      "monitor", "investigate", "bet", "desk", "apk", "re", "pump", "finance", "weather", "tz", "email", "gmail", "mail", "inbox", "routine", "home", "store", "track", "untrack", "trip", "travelclient", "contract", "legal", "contracts", "regwatch", "uprofile", "vnote", "match", "cgroup", "challenge", "room", "scamcheck", "passport", "truecost", "value", "valuepick", "watch", "send", "leads", "panel",
      "connectors", "provision", "pair", "owner", "sudo", "massreact", "pollvote", "chfollow", "notion", "gcal", "trello", "exness", "stripe",
      "guardrails", "content",
      "tempmail", "checkmail", "vnumber", "checksms", "browser", "paper", "backup"]),
    ("platform control — updates & restarts",
     ["update", "pull", "deploy", "restart"]),
    ("groups & communities",
     ["rules", "member", "glog", "gsettings", "strikes",
      "recheck", "joinq", "console", "undonotice",
      "flag", "rep", "repclear", "repdismiss",
      "balance", "mine", "work", "pay", "gshop", "quests",
      "treasury", "deposit", "withdraw", "rob", "bail", "coinflip",
      "top", "tax", "grant", "drop", "claim",
      "property", "buyproperty", "sellproperty", "rent",
      "listproperty", "auctions", "mortgage", "insure",
      "gear", "heist", "shoot", "kill", "assassinate", "sabotage",
      "gang", "war", "police", "bounty", "claimdrop",
      "poll", "vote", "milestones", "contributions", "boards",
      "webshot",
      "mute", "unmute", "ban", "unban", "kick", "warn", "pin",
      "unpin", "purge", "report",
      "cleanup", "rmdeleted", "rminactive",
      "tag", "bansender", "unbansender", "reactwipe", "boosters",
      "party", "ajo"]),
    ("discovery",
     ["list", "commands", "menu", "help"]),
]

#: Septorch-style emoji headers for the menu groups
_GROUP_EMOJI = {
    "her day — state & control": "🎛️",
    "search & research": "🔍",
    "wisdom keeper — corpus · history · practice": "📿",
    "building for real — code & missions": "🏗️",
    "media system — music · playback · video · podcast": "🎬",
    "anime — lookup · schedules · pictures": "🎌",
    "memory & thinking": "🧠",
    "games — 41, DM + group, start them directly": "🎮",
    "voice & vision": "🎙️",
    "tools & automation": "⚙️",
    "platform control — updates & restarts": "🔄",
    "groups & communities": "👥",
    "discovery": "🧭",
}

#: quick aliases → the group they filter to
_LIST_GROUP_ALIASES = {
    "status": "her day — state & control",
    "day": "her day — state & control",
    "control": "her day — state & control",
    "search": "search & research",
    "research": "search & research",
    "net": "search & research",
    "building": "building for real — code & missions",
    "build": "building for real — code & missions",
    "code": "building for real — code & missions",
    "memory": "memory & thinking",
    "memories": "memory & thinking",
    "think": "memory & thinking",
    "voice": "voice & vision",
    "vision": "voice & vision",
    "tools": "tools & automation",
    "automation": "tools & automation",
    "media": "media system — music · playback · video · podcast",
    "anime": "anime — lookup · schedules · pictures",
    "manga": "anime — lookup · schedules · pictures",
    "music": "media system — music · playback · video · podcast",
    "audio": "media system — music · playback · video · podcast",
    "wisdom": "wisdom keeper — corpus · history · practice",
    "games": "games — 41, DM + group, start them directly",
    "predict": "games — 41, DM + group, start them directly",
    "discovery": "discovery",
}

#: one-line "what it does" per command. Explicit entries win; kinds with
#: no explicit entry are backfilled from COMMAND_DETAILS by the loop
#: after the COMMAND_DETAILS auto-fill, so the two tables stay in sync
#: by construction.
LIST_ONELINERS: dict[str, str] = {
    "status": "her live state — platforms, budget, goals, loop",
    "ping": "this message's journey — arrival → receipt → processing → sent, plus her condition",
    "platforms": "which chat platforms are running",
    "mood": "set or read her mood (label or dim=NN)",
    "stage": "relationship stage (committed|dating|…)",
    "model": "switch the brain live: /model [provider [fallbacks]]",
    "providers": "probe every provider: live, cooling down, or why it fails",
    "say": "send as her (/say platform:chat text) · voice note (/say <voice> <text>) · rich voice (/say [direction] <text>)",
    "power": "power mode on/off/status (key-gated)",
    "proposals": "actions waiting for your approval",
    "approve": "approve a pending proposal",
    "deny": "deny a pending proposal",
    "start": "hot-start a platform this session",
    "stop": "hot-stop a platform this session",
    "quit": "shut the system down",
    "mode": "autonomy mode: off | suggest | auto",
    "agent": "agent modes: adaptive | autonomous | planning (combinable)",
    "chatmode": "per-chat mode + feature toggles: /chatmode games-only | games-off | games-on | quiet | normal | status | list | feature <games|music|voice|images|proactive|games.<game>> <on|off>",
    "style": "mined style profile: /style shows it, /style <field> <value> tunes it, /style reset clears overrides",
    "doctor": "system health: registry + config + stores + adapters + providers",
    "profile": "profile-aware runtime: detected environment + every tuned knob",
    "update": "self-update from chat: gate + checkpoint + pull + restart",
    "pull": "git pull --ff-only, no restart",
    "deploy": "deploy this code: gate + checkpoint + pull + restart",
    "restart": "checkpoint missions + restart, no pull",
    "search": "quick research + cited summary",
    "structure": "turn an objective into a structured mission brief",
    "book": "BookForge: write a real book → PDF → send",
    "books": "story reader + continuer: /books read <url|slug> | continue <slug> [n] | sources",
    "decode": "identify + decode anything: /decode <data|file|hash>",
    "investigate": "one-pass investigation: decode → crack → OSINT → knowledge graph",
    "cookies": "YouTube cookie file status — needed when YouTube blocks downloads",
    "cookielab": "CookieLab: analyze & handle cookie headers",
    "cipher": "real crypto: AES-256 sealed blobs + classic ciphers + vault",
    "monitor": "watch a target on a cadence, alert on change",
    "workspace": "the VCPU farm: status/scale/up/down/pause/resume",
    "searchdeep": "deep multi-query research (power mode)",
    "searchleads": "legit paid-task platforms report",
    "money": "money-making opportunities hunter",
    "searchhist": "your recent research runs",
    "research": "web research on any topic + the always-on research loop (status/ideas/approve)",
    "news": "fetch + summarize the feeds",
    "translate": "translate text, a replied-to message, or a link — /translate [to <lang>] <text>",
    "tl": "alias of /translate",
    "steal": "claim replied media into her GIF/sticker library — /steal [tags] as a reply",
    "sticker": "make a sticker from text/emoji, or a replied image/video",
    "convert": "convert replied media — /convert sticker-to-mp4 | gif-to-mp4 | mp4-to-gif | mp4-to-mp3 | mp3-to-video | image-to-sticker | video-to-sticker",
    "gif": "a GIF for the moment, from her library + providers — /gif <what>",
    "emoji": "search the FULL Unicode emoji table by meaning — /emoji <words> · /emoji make <words> creates a real custom emoji",
    "customemoji": "her own custom emoji set — /customemoji list|add|ref|show|remove",
    "emojiart": "emoji art: pictures as braille dot art / ASCII / mosaics, drawn subjects, her combo library, fancy text — /emojiart draw <subject>",
    "react": "react to the replied message — /react [emoji]",
    "awoo": "reaction GIF: awoos at them — /awoo [@someone] or as a reply",
    "baka": "reaction GIF: calls them a baka — /baka [@someone] or as a reply",
    "bite": "reaction GIF: bites them — /bite [@someone] or as a reply",
    "blush": "reaction GIF: blushes at them — /blush [@someone] or as a reply",
    "bonk": "reaction GIF: bonks them — /bonk [@someone] or as a reply",
    "bored": "reaction GIF: yawns at them — /bored [@someone] or as a reply",
    "bully": "reaction GIF: bullies them — /bully [@someone] or as a reply",
    "cringe": "reaction GIF: cringes at them — /cringe [@someone] or as a reply",
    "cry": "reaction GIF: cries beside them — /cry [@someone] or as a reply",
    "cuddle": "reaction GIF: cuddles them — /cuddle [@someone] or as a reply",
    "dance": "reaction GIF: dances with them — /dance [@someone] or as a reply",
    "facepalm": "reaction GIF: facepalms at them — /facepalm [@someone] or as a reply",
    "feed": "reaction GIF: feeds them — /feed [@someone] or as a reply",
    "glomp": "reaction GIF: glomps them — /glomp [@someone] or as a reply",
    "handhold": "reaction GIF: holds hands with them — /handhold [@someone] or as a reply",
    "handshake": "reaction GIF: shakes hands with them — /handshake [@someone] or as a reply",
    "happy": "reaction GIF: smiles at them — /happy [@someone] or as a reply",
    "highfive": "reaction GIF: high-fives them — /highfive [@someone] or as a reply",
    "hug": "reaction GIF: hugs them — /hug [@someone] or as a reply",
    "kiss": "reaction GIF: kisses them — /kiss [@someone] or as a reply",
    "laugh": "reaction GIF: laughs at them — /laugh [@someone] or as a reply",
    "lick": "reaction GIF: licks them — /lick [@someone] or as a reply",
    "nom": "reaction GIF: noms them — /nom [@someone] or as a reply",
    "pat": "reaction GIF: pats them — /pat [@someone] or as a reply",
    "poke": "reaction GIF: pokes them — /poke [@someone] or as a reply",
    "pout": "reaction GIF: pouts at them — /pout [@someone] or as a reply",
    "shrug": "reaction GIF: shrugs at them — /shrug [@someone] or as a reply",
    "slap": "reaction GIF: slaps them — /slap [@someone] or as a reply",
    "sleep": "reaction GIF: sleeps beside them — /sleep [@someone] or as a reply",
    "smile": "reaction GIF: smiles at them — /smile [@someone] or as a reply",
    "smug": "reaction GIF: looks smug at them — /smug [@someone] or as a reply",
    "stare": "reaction GIF: stares at them — /stare [@someone] or as a reply",
    "thumbsup": "reaction GIF: gives them a thumbs up — /thumbsup [@someone] or as a reply",
    "tickle": "reaction GIF: tickles them — /tickle [@someone] or as a reply",
    "wave": "reaction GIF: waves at them — /wave [@someone] or as a reply",
    "wink": "reaction GIF: winks at them — /wink [@someone] or as a reply",
    "yeet": "reaction GIF: yeets them — /yeet [@someone] or as a reply",
    "joke": "a joke from her sources — /joke",
    "dadjoke": "a dad joke — /dadjoke",
    "progjoke": "a programming joke — /progjoke",
    "chuck": "a Chuck Norris fact — /chuck",
    "fact": "a random fact — /fact",
    "fact2": "the fact of the day — /fact2",
    "catfact": "a cat fact — /catfact",
    "numfact": "a fact about a number — /numfact [n]",
    "advice": "a slip of advice — /advice",
    "quote": "a quote + its author — /quote",
    "dog": "a dog picture, attached — /dog",
    "cat": "a cat picture, attached — /cat",
    "fox": "a fox picture, attached — /fox",
    "meme": "a meme, attached (family-safe draw) — /meme",
    "riddle": "a riddle; the answer is held — /riddle · /riddle answer",
    "8ball": "ask the Magic 8-Ball — /8ball [question]",
    "choose": "she picks one of your options — /choose a or b",
    "activity": "something to do — /activity",
    "pokemon": "a Pokémon record + artwork — /pokemon [name|number]",
    "history": "on this day in history — /history [date]",
    "nasa": "NASA's picture of the day — /nasa",
    "calc": "arithmetic, checked evaluator — /calc 2 + 3 * 4",
    "qcard": "make a quote card image of a message — /qcard Ada | we ship at dawn",
    "canvasquote": "quote card, census skin of /qcard — /canvasquote Ada | the words",
    "iqc": "quote card, census skin of /qcard — /iqc Ada | the words",
    "wndwqute": "quote card, census skin of /qcard — /wndwqute Ada | the words",
    "meme2": "a meme-style caption card — /meme2 Ada | it works on my machine",
    "fakemsg": "a styled fake-message card — /fakemsg ttc Ada | first!",
    "fakettc": "a TikTok-comment style card — /fakettc Ada | this is the one",
    "nokiamsg": "an SMS-chat style card — /nokiamsg Ada | come home early",
    "fakeff": "a game-profile style card — /fakeff ShadowFox | booyah tonight",
    "fakeffduo": "a two-speaker exchange card — /fakeffduo Ada | push mid | Bola | on my way",
    "ffgirl": "a game-profile style card — /ffgirl Luna | clutch or kick",
    "removebg": "cut a photo's background out — reply to it with /removebg",
    "qr": "make a QR code — /qr <text>; decode one — /qr read as a reply",
    "shorturl": "shorten a link, destination shown — /shorturl <url>",
    "unshorten": "see where a link really lands — /unshorten <url>",
    "bible": "look up a passage — /bible John 3:16 · /bible search love",
    "couplepp": "a matched pair of profile pictures from one artwork — /couplepp",
    "pp": "alias of /couplepp",
    "archive": "archive a chat on your account — /archive [chat name]",
    "unarchive": "unarchive a chat — /unarchive [chat name]",
    "mutechat": "mute a whole chat — /mutechat [chat name]",
    "unmutechat": "unmute a chat — /unmutechat [chat name]",
    "readchat": "mark a chat read — /readchat [chat name]",
    "unreadchat": "mark a chat unread (WhatsApp; Telegram can't) — /unreadchat",
    "star": "star the replied message (WhatsApp) — /star as a reply",
    "unstar": "unstar the replied message — /unstar as a reply",
    "forward": "forward the replied message to another chat — /forward <chat> as a reply",
    "unsend": "delete your own message for everyone — /unsend as a reply",
    "edit": "alias of /editmsg — /edit <new text> as a reply",
    "editmsg": "edit your own message — /editmsg <new text> as a reply",
    "antiedit": "edit recovery: originals of edited DMs, held for you — /antiedit [on|off]",
    "block": "block a person on your account — /block <number/@user>",
    "unblock": "unblock a person — /unblock <number/@user>",
    "anticall": "reject incoming WhatsApp calls as you — /anticall [on|off]",
    "disappear": "a chat's disappearing-messages timer — /disappear 7d",
    "statusreact": "react to contacts' statuses as you — /statusreact [on|off]",
    "viewonce": "see a view-once sent to you — it was kept in your vault; /viewonce [n]",
    "vv": "short form of /viewonce — /vv [n]",
    "poststatus": "post a status/story as you — /poststatus [text]",
    "chanstory": "post to a channel you admin, as you — /chanstory channel [text]",
    "joinlink": "join a WhatsApp group through its invite link, as you — /joinlink link or code",
    "newgroup": "create a Telegram group as you — /newgroup title",
    "newchannel": "create a Telegram channel as you — /newchannel title",
    "leave": "leave a group or channel as you — /leave [chat name]",
    "banchecker": "honest ban signals: your accounts, or a number's registration — /banchecker [number]",
    "gcstatus": "this group's real stats, sources labeled — /gcstatus",
    "statusinfo": "alias of /gcstatus",
    "pollres": "a poll's current results, from the message — /pollres as a reply",
    "tagall": "mention every member (chunked, capped at 100) — /tagall [text]",
    "hidetag": "mention every member, names hidden — /hidetag [text]",
    "antitag": "mass-tags of you flagged to your DM — /antitag [on|off]",
    "pulse": "briefings — schedule + run any slot on demand",
    "osint": "read-only public-intel: reports, campaigns, graph",
    "dns": "DNS records, zero dependencies",
    "scan": "port scan (own infra) · exposure-audit a URL",
    "audit": "security audit of a repo — secrets, deps, patterns + disclosure draft",
    "whois": "domain registration data (RDAP)",
    "ports": "what is listening on this machine",
    "code": "the coding bot — draft→run→fix to a real acceptance",
    "py": "run python sandboxed (sessions: -s name)",
    "devon": "the autonomous dev agent — plans tools, runs, digests, replies",
    "swarm": "parallel devon agents + fusion",
    "task": "queued instructions she runs and reports on",
    "gen": "generate a validated automation script of a kind (bare /gen lists the kinds)",
    "data": "training data: mine / list / fetch HF sets",
    "evolve": "self-improvement, test-gated (propose|apply|revert|publish)",
    "upgrade": "research→approve→evolve: review tickets, preview patches, approve/deny",
    "arena": "content arena: research, stream, review, approve builds",
    "trial": "ONE trial-account plan, stored encrypted",
    "remember": "store something in long-term memory",
    "recall": "what she remembers (top 5)",
    "forget": "delete a memory",
    "memories": "trust view — people tracked, memory counts, recent contact",
    "spend": "log a spend, e.g. /spend 5k on transport",
    "budget": "set/list monthly budgets, e.g. /budget set food 50k",
    "spending": "spending summary vs budgets (/spending week|month)",
    "mandate": "spending mandates: /mandate issue <scope> <per-txn> [per-day] [days] | list | revoke <id> | stop all",
    "train": "training plans: /train plan <goal> [equipment] [weeks] | today | readiness | log [completed|skipped] [rpe] | list",
    "form": "form check: /form analyze <video> <exercise> | gait <video> | movements | apply <id>",
    "film": "film study: /film analyze <video> <game> [exercise] | games | fingerprint [game] | list [game] | export <id>",
    "graph": "knowledge graph: /graph add <type> <label> | link <from> <to> <type> | disrupt <id> [note] | show [id] | breaks <id> | path | sync",
    "challenge": "photo-proof challenges + streaks + squads: /challenge …",
    "room": "live audio rooms: /room … — host → speakers → hand-raise → listeners",
    "scamcheck": "Nigerian rental scam check — six automated checks",
    "passport": "verify-once rental passport — reusable tenant profile (owner only)",
    "truecost": "true rental move-in cost from listing text (owner only)",
    "value": "DIY property valuation — value range with confidence band + comparables",
    "valuepick": "refine a value estimate from picked comparables",
    "audio": "transcript-as-timeline audio editing (owner only)",
    "overview": "interactive audio overviews — the podcast you can talk to (owner only)",
    "character": "conversational story characters — talk to them in character (owner only)",
    "charagent": "persistent character agents — converse, remember, play games",
    "audiobook": "EPUB → audiobook, one click, with store disclosure (owner only)",
    "watch": "persistent match alerts — saved listing searches",
    "aeo": "AEO/GEO visibility tracking — share of answer (owner only)",
    "send": "self-hosted send layer — campaigns, queue, bounce handling (owner only)",
    "leads": "comment → DM → lead loop — keyword triggers, lead CRM (owner only)",
    "brief": "SERP-derived content brief — keywords, questions, angles (owner only)",
    "content": "brief-first pipeline — score, predict, schedule (owner only)",
    "guardrails": "rule-based ad-spend guardrails — pause/scale rules, mandate-gated (owner only)",
    "competitor": "no-access competitor intel — pillars, cadence, engagement, AEO face-off (owner only)",
    "cost": "transparent spend — 'this answer cost $X' footer, today's spend, research budgets (owner only)",
    "think": "explicit multi-step reasoning with the full trace",
    "benchmark": "how sharp the system is right now (reasoning|planning|tool_use|self_correction)",
    "redteam": "attack my own loop in a sandbox and report the holes",
    "predict": "prediction pit — call a d100 roll, win 2x (unlock: beat challenges)",
    "miniapp": "group mini-apps: polls, quizzes, shared expenses, RSVPs + carpool/rota/market/league/split",
    "party": "group parties: /party new <title> | start | request | vote | trivia | wall | end (+ quick)",
    "ajo": "rotating savings circles: /ajo new <amount> <name> | join | pay | payout | balance",
    "panel": "persistent live-data panels: track fitness, finance, … and ask questions",
    "game": "the social game engine — 41 games, DM + group + channel, with economy and leaderboards",
    "math": "math class — adaptive maths problems, duels, streaks, leaderboard",
    "skill": "learnable battle skills — martial arts for the arena (/skill learn <name>, /skill combos)",
    "title": "earnable titles — flair with battle buffs (/title set <name>)",
    "stats": "RPG attributes — strength/stamina/mana/intelligence (/stats)",
    "gift": "send coins, gear, or items to another player (/gift @name 100)",
    "mind": "the core mind — routes a natural-language goal to the right organ (inspectable)",
    "wordchain": "word chain — last letter becomes first",
    "hangman": "hangman — guess the word before the board is full",
    "numberguess": "number guess — narrow the range until you hit it",
    "two_truths": "two truths and a lie — vote for the lie",
    "wyrr": "wyrr — the house poses, the table picks",
    "spy": "spy — everyone gets a word but one of you lies",
    "auction": "auction — bid points for the block",
    "trivia": "trivia royale — timed trivia showdown",
    "mafia": "mafia — town vs mafia with a night phase",
    "king": "king of the hill — climb and defend",
    "story": "story chain — one sentence each, the story grows",
    "rpg": "rpg adventure — a shared dungeon crawl",
    "shop": "shop game — the market turns into a game",
    "duel": "quiz duel — one-on-one rapid fire",
    "case": "the case — a cooperative investigation",
    "world": "world — a continuing town that just grows",
    "escape": "escape room — 4 locks, break out together",
    "political": "political — 3 elections, be mayor (groups)",
    "tts": "speak text, send as a voice note",
    "speak": "neural voice note of your text",
    "bet": "sports bet analyst: ensemble ML odds analysis + Kelly stakes",
    "desk": "betting desk: crash pattern lab, verdicts library, timed signal calls",
    "apk": "ApkForge: land, dissect, learn from and diff APKs",
    "re": "Reverse engineering: evidence-led static inspection of binaries",
    "pump": "pump.fun intelligence desk: dossiers, de-rug gate, self-graded signal calls",
    "weather": "live weather, alerts, USA situations overview",
    "tz": "timezone conversion + your labeled local time",
    "stt": "transcribe an audio file",
    "look": "screen-reader analysis of a screenshot (sees pixels)",
    "imggen": "Devon's own image studio: native diffusion, upscale, training",
    "image": "look up an image (format, dims, seen-before) — or generate one from a prompt",
    "lens": "reverse image search",
    "schedule": "cron-style in-process jobs",
    "db": "inspect the database (tables|schema|query|counts)",
    "artifact": "saved artifacts: list, info, or send one to this chat",
    "api": "external API connectors (weather, fx, github, …)",
    "connectors": "manage the service connectors: list, status, connect",
    "shorts": "short-form content empire: make, queue, post vertical videos",
    "notion": "Notion databases, rows, and page creation",
    "gcal": "Google Calendar agenda and event creation",
    "trello": "Trello boards, lists, cards — create cards",
    "proxy": "proxy lab: status|test|set|scrape|refresh|health|pool|pick|affinity|rotate|schedule|ssh",
    "record": "capture actions into a macro",
    "macro": "replay a recorded macro",
    "file": "send a file to any live chat",
    "publish": "md→pdf/html and send",
    "notify": "recent alerts",
    "proactive": "proactive push sends: switches + delivery states",
    "mission": "mission progress + ETA + stall reasons",
    "email": "mailbox (Gmail): inbox/search/read/thread + triage + draft queue; sends need your explicit 'send it'",
    "gmail": "alias of /email — the owner's mailbox over Gmail",
    "mail": "alias of /email — the owner's mailbox over Gmail",
    "inbox": "alias of /email — bare /inbox lists the inbox",
    "features": "feature toggles (arena, vision, search, …)",
    "list": "this catalog — every executable command, categorized",
    "help": "the full help: catalog, per-command pages, topics",
    "music": "compose a real song: lyrics + chords + playable MIDI",
    "play": "media player: queue + transport (mpv when installed)",
    "anime": "anime lookup — record, score, episodes, cover (AniList + MAL)",
    "manga": "manga lookup — record, score, chapters, cover",
    "anichar": "anime character lookup — who they are, their picture",
    "charpic": "a random anime character picture",
    "staff": "staff lookup — voice actors, directors, their work",
    "studio": "studio lookup — record + best-known works",
    "episodes": "a title's episode list + guide",
    "animechars": "a title's cast",
    "trailer": "a title's trailer (watch it via /video)",
    "streaming": "legal watch links on a title's record",
    "animereview": "scores + community reviews for a title",
    "genres": "the genre list on file",
    "genre": "top anime in one genre",
    "topanime": "highest-scored anime of all time",
    "topmanga": "highest-scored manga of all time",
    "popular": "most popular anime (or manga) right now",
    "trending": "trending anime (or manga) right now",
    "airing": "airing now, with next-episode countdowns",
    "upcoming": "announced anime, not yet out",
    "season": "one season's anime",
    "topchars": "most-loved characters",
    "anischedule": "the week's airing grid (WAT)",
    "nextep": "when a title's next episode airs (WAT + countdown)",
    "recommend": "more like a title, from its record",
    "randomanime": "a random anime record",
    "randommanga": "a random manga record",
    "randomchar": "a random character record",
    "animequote": "an anime quote, attributed",
    "favorite": "your saved anime favorites (your list only)",
    "whatanime": "name the anime in a replied frame (episode + timestamp)",
    "waifu": "a waifu picture",
    "neko": "a neko picture",
    "kitsune": "a kitsune picture",
    "husbando": "a husbando picture",
    "video": "find videos across the web (ranked, enriched) + download",
    "director": "Devon directs: coverage → breakdown + shot design → shoot → QA",
    "exec": "run code sandboxed: py/js/bash/c/… with real output",
    "zip": "archives: create/list/info/extract (safe against zip-slip)",
    "deliver": "create-and-deliver: styled report → zip → send to chat",
    "apps": "scaffold + serve runnable apps: static|flask|fastapi|express|react|cli",
    "hub": "one-call media: song / video / podcast pipelines, end to end",
    "podcast": "find → download → transcribe → chapters, transcript saved",
    "fix": "run code, model rewrites failures until it's green (CI loop)",
    # reachability wave — one-liners for the nine new commands (these
    # feed COMMAND_DETAILS via the auto-fill loop) plus the previously
    # weak/missing ones (alert/balances/goal/insights had 'run /x.' text).
    "tempmail": "temp inbox: /tempmail new gets an address (mail.tm → 1secmail → guerrilla), per-user",
    "checkmail": "read your temp inbox: /checkmail [read <n> | code | forget]",
    "vnumber": "temp number: /vnumber new [country] gets a virtual number for SMS codes, per-user",
    "checksms": "read SMS on your temp number: /checksms [code | forget]",
    "browser": "the browser tool in chat: /browser open|text|links|forms|headings <url> | takeover [url] | run <url> [goal] (owner-only, background job) | run status [job] | watch [session] — attach a live view to a running browser session (owner-only)",
    "provision": "self-provisioning: /provision [service] — Devon fetches a missing API key herself (login or account creation; key validated + vaulted, request retried); /provision login <service> <username> hands her an existing login",
    "pair": "pairing-with-code pool: /pair whatsapp <number> [label] — pair a WhatsApp account by 8-char code (own bridge session each); /pair telegram <number> pairs Telegram by login code into the same pool; /pair lists the pool with Registered/Connected/Dropped counts (owner only)",
    "sudo": "sudo rung (§19): /sudo — who holds sudo, in plain words; /sudo add|remove <platform> <id> grants or drops it through the owner grammar (owner only)",
    "owner": "the owner system: /owner — the verification card (bound ids per platform + how each was proved + the seal's state); /owner status — uptime, version, pool counts, queues, last briefing; bind/unbind/sudo/deputy/mark edits and the sealed rebind (owner only, DM only — never editable from a group)",
    "massreact": "mass react: /massreact <post link> | <emoji> [| count] [| live] — one reaction on a post from every connected account in your pool (Telegram + WhatsApp channels); arm it once first, first run dry-runs (owner only)",
    "pollvote": "mass vote: /pollvote <poll link> | <option> [| count] [| live] — vote in a poll from every connected Telegram account (MTProto; bots can't vote); arm-gated, first run dry-runs (owner only)",
    "chfollow": "channel follow: /chfollow <channel link> [| live] — every connected account in your pool follows the channel; idempotent; arm-gated, first run dry-runs (owner only)",
    "paper": "paper trading: /paper start <symbol> [market] [capital] | status | stop | list (simulated, never real)",
    "people": "the people spine: /people list | show <name> | add <name> | note <name> <text>",
    "commitments": "the obligation layer: /commitments list | add <what> [by <deadline>] | kept|broken|cancel <id> | overdue | due | reliability | verify",
    "decisions": "the decision journal — calibrate your own calls: /decisions add <decision> [| chosen | prediction | conf%] | resolve <id> [good|bad] <outcome> | list | review [weekly|monthly] | calibration",
    "iddocs": "identity document expiry tracker (owner-only): /iddocs list | add <type> <expiry YYYY-MM-DD> <vault:ref> [country] | expiring [days] | overdue | renew <id> — vault references only, never document numbers",
    "focus": "focus sessions: /focus start [label] | pause | resume | stop [note] | status | today | week | month",
    "backup": "backup health (read-only): /backup status | verify — generations, manifest, DB integrity",
    "alert": "price/money alerts: /alert add|list|remove|check|snooze|sync-bills",
    "balances": "unified money view across rails: /balances [rail…]",
    "goal": "money goals: /goal add|list|done|snooze|remove",
    "insights": "native money analytics: /insights [days]",
    "caption": "burn AI subtitles into a video: /caption [style] (attach or path)",
    "checkpoint": "snapshot the coding workdir: /checkpoint [label]",
    "checkpoints": "list saved coding checkpoints: /checkpoints",
    "clonevoice": "clone a voice from a voice note: /clonevoice <name> (attach the note)",
    "dislike": "last production missed — with notes: /dislike [notes…]",
    "distribute": "release walk-through: /distribute [song.wav] | legal",
    "dj": "radio show with voice breaks: /dj [trending|<genre>]",
    "download": "any-URL media downloader: /download <url> [audio]",
    "tiktok": "TikTok downloader (no-watermark route first, labeled honestly): /tiktok <url> [audio]",
    "ttdl": "alias of /tiktok: /ttdl <url> [audio]",
    "instagram": "Instagram reel/post downloader (public; login wall named): /instagram <url>",
    "spotify": "identify a Spotify link, fetch the full track from a real source: /spotify <url>",
    "aio": "all-in-one downloader — any supported link, routed by platform: /aio <url> [audio]",
    "exness": "Exness trading: /exness balance|positions|price|buy|sell|close",
    "heal": "self-healing loop: /heal [--dry] | /heal <cmd> — diagnose + auto-fix broken commands",
    "immune": "defensive immune system: /immune status|scan|antibodies|quarantine|disarm|arrest|fever|drift|casefile",
    "impossible": "Impossible Made Possible: /impossible solve <problem> | status | attempt | dossier | queue | charter | lattice | examples",
    "like": "last production was good: /like [notes…]",
    "novel": "webnovel tracker: /novel follow <url|title> | list | read <slug> [ch] | continue <slug> [n]",
    "produce": "compose original music, taste-aware: /produce [spotify-url|vibe words…]",
    "rewind": "restore a coding checkpoint: /rewind [n]",
    "sham": "reply to an audio/voice note to identify the song: /sham",
    "musicnote": "answer the music voice-note ask: /musicnote find | create",
    "sing": "sing a melody: /sing <melody> (C4:0.5:hello …)",
    "stripe": "Stripe payments: /stripe balance|customers|charges|link",
    "tour": "smoke-test every command: /tour | /tour <cmd> deep-probes one",
    "vision": "analyze an image: /vision [question] (attach or path/URL)",
    "voices": "list every voice, cloned vs built-in: /voices",
    "pvp": "player-vs-player arena duels: /pvp [opponent]",
    "raid": "raid boss — the table vs one big boss: /raid",
    "commands": "alias of /list — every executable command, categorized",
    "menu": "alias of /list — every executable command, categorized",
    "wis": "alias of /wisdom",
    "wisdom": "WisdomKeeper: ask the esoteric corpus (answer + provenance), browse the history timeline, or run a guided breathing session",
}


#: Fill COMMAND_DETAILS for kinds that only have a one-liner or a game
#: blurb. This loop MUST run after LIST_ONELINERS is defined (it reads
#: it) — it used to sit above LIST_ONELINERS, which made the whole
#: module unimportable (NameError at import time).
for _kind in CONTROL_COMMANDS:
    if _kind not in COMMAND_DETAILS:
        _g = _GAME_HELP.get(_kind)
        if _g is not None:
            COMMAND_DETAILS[_kind] = {
                "what": _g,
                "usage": f"/{_kind} [bet] — starts it; /game controls the session",
                "example": f"/{_kind}",
                "related": "/game /list",
            }
        else:
            COMMAND_DETAILS[_kind] = {
                "what": LIST_ONELINERS.get(_kind, f"run /{_kind}."),
                "usage": f"/{_kind}",
                "example": f"/{_kind}",
                "related": "/list /help",
            }

#: Normalize COMMAND_DETAILS: every page carries what/usage/example/
#: related, so /help <command> renders the same shape for every command.
#: Explicit non-blank values win; missing or blank fields are defaulted
#: (a few entries carried "related": "" — an empty related line).
for _kind in CONTROL_COMMANDS:
    _d = COMMAND_DETAILS[_kind]
    if not str(_d.get("related") or "").strip():
        _d["related"] = "/list /help"
    if not str(_d.get("usage") or "").strip():
        _d["usage"] = f"/{_kind}"
    if not str(_d.get("example") or "").strip():
        _d["example"] = f"/{_kind}"

#: Backfill LIST_ONELINERS from COMMAND_DETAILS so /list can never show
#: a command without its one-liner: the two tables are one source of
#: truth by construction, not by a comment asking humans to keep them
#: in sync. Explicit one-liners above win; only missing kinds are filled.
for _kind in CONTROL_COMMANDS:
    if _kind not in LIST_ONELINERS:
        LIST_ONELINERS[_kind] = COMMAND_DETAILS[_kind]["what"]


def capability_brief(*, per_group: int = 5) -> str:
    """Her real capabilities — rendered from THIS catalog.

    The ground truth the brain answers "what can you do?" from
    (register #34): the same registry /help renders — the grouped
    systems (LIST_GROUPS), each with real command names and their
    catalog one-liners, and the command total computed from
    CONTROL_COMMANDS, never a hardcoded count.  Compact on
    purpose: a map for the prompt, not the manual — /help carries
    the full text.  When the catalog grows, this grows with it;
    there is no second list to drift.
    """
    total = len(CONTROL_COMMANDS)
    lines = [f"{total} commands across {len(LIST_GROUPS)} systems:"]
    for title, kinds in LIST_GROUPS:
        picks: list[str] = []
        for kind in kinds[:per_group]:
            what = str((COMMAND_DETAILS.get(kind) or {}).get("what")
                       or "").strip()
            what = what.split(" — ")[0].split(". ")[0].rstrip(".")
            picks.append(f"/{kind} — {what}" if what else f"/{kind}")
        more = (f" (+{len(kinds) - len(picks)} more)"
                if len(kinds) > len(picks) else "")
        lines.append(f"- {title}: " + "; ".join(picks) + more)
    lines.append("/help renders the full catalog; "
                 "/help <command> details any one of them.")
    return "\n".join(lines)


def list_catalog(topic: str = "", *, public_only: bool = False) -> str:
    """/list — every executable chat command, cleanly categorized (wave 68).

    No topic: the full categorized catalog.  A topic (group name or
    alias): just that group.  Every line is ``/cmd — what it does``.

    ``public_only`` renders the gated public surface's version: only
    PUBLIC_COMMANDS are listed, and groups with no public commands
    left in them drop out (including from the "available" pointer).

    ``topic`` must be a string (``None``/``""`` mean the full catalog);
    anything else raises :class:`TypeError`.
    """
    if topic is not None and not isinstance(topic, str):
        raise TypeError(f"list_catalog expects str, got {type(topic).__name__}")
    t = (topic or "").strip().lstrip("/").lower()
    groups = LIST_GROUPS
    if t:
        want = _LIST_GROUP_ALIASES.get(t, t)
        groups = [(g, ks) for g, ks in LIST_GROUPS
                  if t in g.lower().replace(" — ", " ") or g.lower() == want
                  or any(t == k for k in ks)]
    from .style import section, cmd, cta, escape
    lines = [f"🥷 <b>Devon</b> — commands (tap any {cmd('/command')} to run it):"]
    shown = 0
    for group, kinds in groups:
        renderable = [k for k in kinds if k in CONTROL_COMMANDS]
        if public_only:
            renderable = [k for k in renderable if k in PUBLIC_COMMANDS]
            if not renderable:
                continue
        emoji = _GROUP_EMOJI.get(group, "•")
        lines.append("")
        lines.append(section(emoji, group))
        for k in renderable:
            one = LIST_ONELINERS.get(k, COMMAND_DETAILS.get(k, {}).get('what', ''))
            lines.append(f"  {cmd('/' + k)} — {escape(one)}")
            shown += 1
    if not shown:
        if public_only:
            avail = " | ".join(
                g for g, ks in LIST_GROUPS
                if any(k in CONTROL_COMMANDS and k in PUBLIC_COMMANDS
                       for k in ks))
            return f"no group '{t}' — available: {escape(avail)}"
        groups_now = [g for g, _ in groups]
        all_groups = " | ".join(dict.fromkeys(_LIST_GROUP_ALIASES.values()))
        return (f"no group '{t}' — available: {escape(all_groups)}")
    lines += [
        "",
        f"📦 <b>{shown} commands</b>",
        f"  {cmd('/help')} &lt;command&gt; — full page  ·  {cmd('/help')} topics — subject guides",
        cta("try /game arena or /search to start"),
    ]
    return "\n".join(lines)


def detailed_help(topic: str = "", *, public_only: bool = False) -> str:
    """The detailed help for chat (and ``nm help``).

    * no topic — the catalog: every command grouped, topic pages listed.
    * ``/help <command>`` — what it does, exact usage, a real example,
      related commands.
    * ``/help budget|goals|builds|skills|missions|modes`` — topic pages.
    * unknown — fuzzy match or a pointer, never a dead end.

    ``public_only`` renders the gated public surface's version: the
    overview lists PUBLIC_COMMANDS only, and pages for owner-only
    commands / owner-system topic pages are not served.

    ``topic`` must be a string (``None``/``""`` mean the catalog); anything
    else raises :class:`TypeError`.
    """
    if topic is not None and not isinstance(topic, str):
        raise TypeError(f"detailed_help expects str, got {type(topic).__name__}")
    t = (topic or "").strip().lstrip("/").lower()
    if not t:
        return _detailed_overview(public_only=public_only)
    if t in COMMAND_DETAILS:
        if public_only and t not in PUBLIC_COMMANDS:
            return _public_no_help_page(t)
        from .style import cmd, escape
        d = COMMAND_DETAILS[t]
        lines = [f"📖 <code>{escape(d['usage'])}</code>",
                 escape(d['what']),
                 "",
                 f"💡 try: {cmd(d['example'])}"]
        if d.get("related"):
            lines.append(f"🔗 related: {cmd(d['related'])}")
        lines.append("")
        lines.append(f"{cmd('/help')} — catalog  ·  {cmd('/help')} topics — subject guides")
        return "\n".join(lines)
    if t in _TOPIC_PAGES:
        if public_only:
            return _public_no_help_page(t)
        return _TOPIC_PAGES[t]
    cands = sorted(k for k in COMMAND_DETAILS
                   if t in k or k in t or t in k.replace("_", " "))
    if public_only:
        cands = [c for c in cands if c in PUBLIC_COMMANDS]
    if len(cands) == 1:
        return detailed_help(cands[0], public_only=public_only)
    if cands:
        return ("did you mean: "
                + ", ".join(f"/help {c}" for c in cands[:8])
                + "\ncatalog: /help")
    if public_only:
        return _public_no_help_page(t)
    return (f"no help page for '{t}' — /help <command> for any command "
            "(e.g. /help devon), or /help budget|goals|builds|skills|"
            "missions|modes for topics.")
