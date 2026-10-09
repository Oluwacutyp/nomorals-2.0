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
    /model [provider [fallback]] switch the model live (no restart, persists)
    /say <platform:chat> text    send a message as her, through the gateway
    /proposals                   pending autonomous messages
    /approve <id>                release a proposal
    /deny    <id>                kill a proposal
    /stage                       show relationship stage + trust
    /stage <stage>               advance/regress explicitly
    /power on <key>              unlock power mode (owner key)
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

__all__ = ["CONTROL_COMMANDS", "COMMAND_DETAILS", "ControlCommand",
           "parse_control", "PLATFORMS", "help_text", "help_menu", "whatsapp_menu",
           "detailed_help",
           "list_catalog", "LIST_GROUPS", "LIST_ONELINERS", "GAME_COMMANDS"]

PLATFORMS = ("telegram", "discord", "whatsapp", "local")

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

#: kind -> (min_args, max_args) — used by /help and by validation.
CONTROL_COMMANDS: dict[str, tuple[int, int]] = {
    "help": (0, 1),
    "status": (0, 0),
    "platforms": (0, 0),
    "start": (1, 1),
    "stop": (1, 1),
    "mood": (0, 10),  # label, or several dim=NN pairs
    "mode": (1, 1),
    "model": (0, 6),  # provider + up to 5 fallback providers
    "providers": (0, 0),  # /providers — probe the whole chain, live or why not
    "say": (2, None),
    "proposals": (0, 0),
    "approve": (1, 1),
    "deny": (1, 1),
    "stage": (0, 1),
    "power": (1, 2),
    "quit": (0, 0),
    "profile": (0, 0),      # /profile — the profile-aware runtime (env + tuned knobs)
    # search engine (Telegram / WhatsApp / console — same commands everywhere)
    "search": (1, None),     # /search <query> — free text, no length cap
    "searchdeep": (1, None), # /searchdeep <query>  (power mode)
    "searchleads": (0, 0),   # legit paid-task platform report
    "money": (0, None),      # money opportunities hunter: scan|list|new|profile
    "searchhist": (0, 1),    # /searchhist [n]
    # BookForge: write a real book → PDF → send when done
    "book": (0, None),       # /book <topic> [chapters] | status | list | build <slug> | send <slug> <p> <c>
    "novel": (0, None),      # /novel follow <url|title> | list | read <slug> [ch] | continue <slug> [n]
    # WisdomKeeper: esoteric corpus Q&A, history timeline, guided practice
    "wisdom": (0, None),     # /wisdom ask <q> | practice list|<id>|stop | timeline [t] | compare <t> | status
    "wis": (0, None),        # alias of /wisdom
    # Universal Decoder: identify + decode anything, send binary results
    "decode": (0, None),     # /decode <data> | /decode file:<path> | /decode hash <digest>
    # Cookie analysis & handling (the CookieLab)
    "cookies": (1, None),    # /cookies <header> | /cookies file:<path> | /cookies ingest <header>
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
    # expansion wave
    "game": (0, 12),         # /game [list|<name>|quit|leaderboard|stats|shop|balance]
    "npc": (0, None),        # /npc list | talk <name> <msg> | mood <name> — the living cast
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
    "blackjack": (0, 0), "roulette": (0, 0), "slots": (0, 0),
    # inbox (wave F2): async, one message per turn, no clock
    "gomoku": (0, 0), "reversi": (0, 0), "checkers": (0, 0),
    # puzzles (gamesbooks-1.0)
    "sudoku": (0, 0), "anagram": (0, 0), "cryptogram": (0, 0),
    # wave 87: the Core Mind — manual override over natural-language routing
    "mind": (0, None),       # /mind [status|clear|<goal>] — the core mind
    "news": (0, 2),          # /news [run|status]
    "research": (0, 2),      # /research [run [domain]|status]
    "code": (1, None),       # /code <what to build> — the coding bot
    "checkpoint": (0, None),  # /checkpoint [label] — snapshot the coding workdir
    "rewind": (0, 1),      # /rewind [n] — restore the nth-latest checkpoint
    "checkpoints": (0, 0),  # /checkpoints — list saved checkpoints
    "py": (1, None),         # /py <python code> — run in the sandbox (-s/-r sessions)
    "remember": (1, None),   # /remember <text> [kind] [tags:a,b]
    "review": (0, 1),       # /review [days] — "how did I do?" representation review
    "tutor": (0, None),       # /tutor [guide me] <topic> | answer <text> | hint | stop | status
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
    "research": (0, None),    # /research <legal question> — grounded RAG with confidence + traceable citations
    "contracts": (0, None),    # /contracts … — contract portfolio: obligations, renewals, SLA tracking
    "regwatch": (0, None),     # /regwatch … — regulatory change monitoring (CBN/SEC/NDPA/FIRS)
    "uprofile": (0, None),     # /uprofile … — prompt-scaffolded profiles + element-level likes (gig/community/business)
    "vnote": (0, None),        # /vnote … — voice notes in chat (audio + searchable transcripts)
    "match": (0, None),        # /match … — curated daily batches + stable two-sided matching
    "cgroup": (0, None),       # /cgroup … — themed groups, events, meetups (#84)
    "challenge": (0, None),    # /challenge … — photo-proof challenges + streaks + squads (#87)
    "room": (0, None),           # /room … — live audio rooms: host → speakers → hand-raise (#107)
    # new layer: voice, scheduler, db, api, vision, swarm
    "tts": (1, None),        # /tts <text> — speak it (sends the audio file)
    "stt": (1, 5),           # /stt <path> — transcribe an audio file
    "look": (1, None),       # /look <path|url> [focus] — screen-reader analysis
    "schedule": (0, None),   # /schedule add|list|rm|enable|disable|run|status
    "db": (0, None),         # /db tables | schema <t> | query <sql> | counts
    "api": (0, None),        # /api list | <connector> [json params]
    "connectors": (0, 2),   # /connectors [list|status|connect] [name] — manage the service connectors
    "notion": (0, None),      # /notion dbs|query|add — Notion reads + page writes
    "gcal": (0, None),        # /gcal [agenda] | add — Google Calendar
    "trello": (0, None),      # /trello boards|lists|cards|add — Trello
    "exness": (0, None),      # /exness balance|positions|price|buy|sell|close — Exness trading
    "stripe": (0, None),      # /stripe balance|customers|charges|link — Stripe payments
    "swarm": (1, None),      # /swarm <goal> [workers] — parallel devon agents
    # power layer: network, proxies, scripts, osint, macros
    "dns": (1, 5),           # /dns <domain> [record type]
    "scan": (1, 6),          # /scan <target> [ports] [banner]
    "whois": (1, 3),         # /whois <domain>
    "ports": (0, 1),         # /ports — what is listening here
    "proxy": (0, 8),         # /proxy status|scrape|test|health|pool|pick|affinity|rotate on|set <url>|ssh <…>
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
    "finance": (0, None),    # /finance quote|analyze|signal|idea|backtest|strategies|compare|watch|doctor
    "weather": (0, None),    # /weather [place] — live conditions + forecast + alerts
    "tz": (0, None),         # /tz [time] — timezone conversion, labeled in owner tz
    "task": (1, None),       # /task add <instruction>|run [id]|list
    "notify": (0, 1),        # /notify [n] — recent alerts
    "proactive": (0, 0),     # /proactive — push-send switches + delivery states
    "email": (0, None),        # /email triage|drafts|send <id>|followups — AI email triage
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
    "list": (0, 1),          # /list [group] — every executable command, categorized
    "commands": (0, 1),      # alias of /list
    # redteam: defensive self-attack suite
    "redteam": (0, 1),       # /redteam [scenario_id] — attack our own loop, report holes
    "predict": (0, 3),       # /predict <stake> <over|under> <1-99> — prediction pit (progression-unlocked)
    "menu": (0, 1),          # alias of /list
    # wave 72 systems: media, execution, archives, builders
    "music": (0, None),      # /music <topic> [style] | styles | song [slug] | bed | full | voices
    "distribute": (0, 1),   # /distribute [song.wav] | legal
    "play": (0, None),       # /play <paths…> | status | queue | pause | …
    "video": (0, None),      # /video <query> | download <url> | platforms
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
        "  /power on <key> | /power off | /power status",
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
        "  — fiction studio (genre engines, novel|serial) —",
        "  /book fiction <premise> --genre <g> --mode <novel|serial>",
        "  /book fiction-write <slug>              next chapter (validated)",
        "  /book fiction-status <slug>             arcs, threads, cast",
        "  — wisdom keeper (corpus · history · practice) —",
        "  /wisdom ask <query>                     ask the corpus (answer + provenance)",
        "  /wisdom practice list | <session-id>    list sessions / start a guided session here",
        "  /wisdom practice stop                   stop the running session",
        "  /wisdom timeline [tradition] | /wisdom compare <topic> | /wisdom status",
        "  — decode & crypto —",
        "  /decode <data> | /decode file:<path>     Universal Decoder (sends binary back)",
        "  /decode hash <digest> | /decode decoders identify hashes / list engine",
        "  /cookies <header> | /cookies ingest <h>  CookieLab: classify, fingerprint, decode",
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
        "  /email triage|drafts|send <id>|followups  AI email triage + draft queue",
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
        "  /tts <text>                             speak it aloud (audio sent back)",
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
        "  /quit",
    ]
)


@dataclass(frozen=True)
class ControlCommand:
    kind: str
    arg: str = ""    #: first argument ("" when there are none)
    tail: str = ""   #: everything after the command word (for multi-word commands)

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
    if kind not in CONTROL_COMMANDS:
        return None  # unknown slash: treat as a normal message, let her answer
    tail = words[1].strip() if len(words) > 1 else ""
    arg = tail.split()[0] if tail else ""
    n = len(tail.split()) if tail else 0
    min_args, max_args = CONTROL_COMMANDS[kind]
    if n < min_args:
        return ControlCommand(kind="error", arg=f"/{kind} needs {min_args} argument(s) — /help {kind} shows the usage")
    if max_args is not None and n > max_args:
        return ControlCommand(kind="error", arg=f"/{kind} takes at most {max_args} argument(s) — /help {kind} shows the usage")
    return ControlCommand(kind=kind, arg=arg, tail=tail)


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

    Parses :data:`_HELP_TEXT` (the same source ``help_text()`` uses, so
    command coverage can never drift) into emoji-headed sections and
    renders them for the target platform — HTML on Telegram,
    ``*bold*``/`` `code` `` on WhatsApp. Never raises.
    """
    try:
        if str(platform or "").strip().lower() in ("whatsapp", "wa"):
            # workstream 3: WhatsApp gets its OWN menu — text-first,
            # numbered, no inline-button assumptions. The Telegram
            # rendering below is untouched.
            return whatsapp_menu()
        from .style import render_menu
        sections: list[tuple[str, str, list[tuple[str, str]]]] = []
        cur_emoji, cur_title = "🎛️", "control commands"
        cur_items: list[tuple[str, str]] = []
        notes: list[str] = []

        def flush() -> None:
            if cur_items or notes:
                items = list(cur_items)
                for n in notes:
                    items.append(("", n))
                sections.append((cur_emoji, cur_title, items))

        for raw in help_text().split("\n"):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("—") and line.endswith("—"):
                flush()
                cur_title = line.strip("— ").strip()
                cur_emoji = _help_section_emoji(cur_title)
                cur_items, notes = [], []
            elif line.startswith("/"):
                parts = re.split(r"\s{2,}", line, maxsplit=1)
                cmd = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
                cur_items.append((cmd, desc))
            elif line.startswith("“") or "→" in line:
                notes.append(line)
            else:
                notes.append(line)
        flush()
        # drop empty-command note rows from the styled render's item list
        clean = [(e, t, [(c, d) for c, d in items if c]) for e, t, items in sections]
        return render_menu(
            clean, platform,
            header="Devon — command menu",
            footer="say it in plain words, or tap a command — /help <command> for detail",
        )
    except Exception:  # noqa: BLE001
        return help_text()


#: WhatsApp-native section curation: (LIST_GROUPS name, display title).
#: Commands still come from LIST_GROUPS at render time — this table only
#: sets the WhatsApp-first order and the short titles. An entry naming a
#: group that no longer exists is skipped; LIST_GROUPS entries missing
#: here are appended at the end in registry order, so the menu can never
#: drift from the registry.
_WHATSAPP_SECTION_ORDER: list[tuple[str, str]] = [
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


def _whatsapp_sections() -> list[tuple[str, str, list[tuple[str, str]]]]:
    """Menu sections for WhatsApp, assembled from the registry at call time.

    The first section comes from the WhatsApp adapter's GROUP_CAPABILITIES
    table (group/community depth over the Baileys bridge); the rest come
    from LIST_GROUPS with one-liners from LIST_ONELINERS/COMMAND_DETAILS
    (the same sources /list renders from). A trailing "more" section
    catches any registered command missing from every section, so
    coverage can never drift. Returns ``[(emoji, title, [(cmd, desc)])]``.
    """
    from . import whatsapp as _wa  # local import: never at module load
    sections: list[tuple[str, str, list[tuple[str, str]]]] = [(
        "💬", "groups & communities",
        [(f'"{say}"', what) for _, say, what in _wa.GROUP_CAPABILITIES],
    )]
    by_name = dict(LIST_GROUPS)
    ordered = [g for g, _ in _WHATSAPP_SECTION_ORDER if g in by_name]
    ordered += [g for g, _ in LIST_GROUPS if g not in ordered]
    titles = dict(_WHATSAPP_SECTION_ORDER)
    for group in ordered:
        title = titles.get(group, group)
        emoji = _GROUP_EMOJI.get(group, "📌")
        items: list[tuple[str, str]] = []
        for kind in by_name[group]:
            if kind not in CONTROL_COMMANDS:
                continue
            one = LIST_ONELINERS.get(
                kind, COMMAND_DETAILS.get(kind, {}).get("what", ""))
            items.append(("/" + kind, one))
        sections.append((emoji, title, items))
    # coverage: every command help_text() knows must appear somewhere
    shown = {cmd for _, _, items in sections for cmd, _ in items}
    missing = sorted(
        kind for kind in CONTROL_COMMANDS
        if f"/{kind}" not in shown and f"/{kind}" in help_text())
    if missing:
        sections.append((
            "📌", "more",
            [("/" + kind,
              LIST_ONELINERS.get(kind,
                                 COMMAND_DETAILS.get(kind, {}).get("what", "")))
             for kind in missing],
        ))
    return sections


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
        want = dict(_WHATSAPP_SECTION_ORDER).get(registry_name, registry_name)
        for i, (_, title, _) in enumerate(sections):
            if title == want:
                return i
    return None


def whatsapp_menu(page: str = "") -> str:
    """WhatsApp-native command menu — text-first, no buttons, no HTML.

    Data-driven, never a pasted copy of the Telegram menu: the
    "groups & communities" section comes from the WhatsApp adapter's
    GROUP_CAPABILITIES table and every other section comes from the
    command registry (LIST_GROUPS / LIST_ONELINERS / COMMAND_DETAILS),
    so it can never drift from what the system actually runs. Only the
    WhatsApp-first order and short titles are curated
    (:data:`_WHATSAPP_SECTION_ORDER`).

    * ``page=""`` — the full menu: a numbered section index, then every
      section fully rendered (the WhatsApp send path chunks long text).
    * ``page="3"`` — just section 3 (the "reply 3" interaction).
    * ``page="voice"`` — the section that title/registry name matches.

    Never raises: junk pages fall back to the full menu; total failure
    falls back to :func:`help_text`.
    """
    try:
        from .style import menu_divider, menu_item, menu_section
        sections = _whatsapp_sections()
        n = len(sections)

        def _section_page(idx: int) -> str:
            emoji, title, items = sections[idx]
            lines = [menu_section(emoji, title, "whatsapp"), ""]
            for command, desc in items:
                lines.append(menu_item(command, desc, "whatsapp"))
            lines.append("")
            lines.append("_reply 0 for the full index · "
                         "/help play for one command's full page_")
            return "\n".join(lines).rstrip()

        idx = _whatsapp_find_section(sections, page)
        if idx is not None:
            return _section_page(idx)

        lines = [menu_section("✨", "Devon — command menu", "whatsapp"), ""]
        lines.append("_text-first — no buttons, no tapping needed_")
        lines.append("")
        for i, (emoji, title, items) in enumerate(sections, 1):
            lines.append(f"{i} · {emoji} *{title.upper()}* ({len(items)})")
        for emoji, title, items in sections:
            lines.append("")
            lines.append(menu_divider("whatsapp"))
            lines.append(menu_section(emoji, title, "whatsapp"))
            for command, desc in items:
                lines.append(menu_item(command, desc, "whatsapp"))
        lines.append("")
        lines.append(f"_reply 1–{n} for just that section · "
                     "say it in plain words · /help play for one command's full page_")
        return "\n".join(lines).rstrip()
    except Exception:  # noqa: BLE001
        return help_text()


# ── detailed help (wave 67) ─────────────────────────────────────────────────
#: Per-command detail: what it does, exact usage, a real example, and
#: related commands.  ``/help <command>`` in chat (and ``nm help <topic>``)
#: renders one of these; ``/help`` alone renders the catalog.
COMMAND_DETAILS: dict[str, dict[str, str]] = {
    "status": {"what": "her live state — platforms, model budget, active goals, loop cadence.",
               "usage": "/status", "example": "/status",
               "related": "/platforms /benchmark"},
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
             "related": "/proposals /power"},
    "model": {"what": "switch the active model live (up to 5 fallbacks); no argument shows the chain.",
              "usage": "/model [provider [fallback …]]", "example": "/model local",
              "related": "/status /providers"},
    "providers": {"what": "probe every registered provider and report live/dead with the reason — "
                          "the diagnostic for a silent brain.",
                  "usage": "/providers", "example": "/providers",
                  "related": "/model /status"},
    "say": {"what": "send a message to a chat as her, from your hands — or speak text as a voice note in a named voice.",
            "usage": "/say <platform:chat> <text> | /say <voice> <text>",
            "example": "/say telegram:123 hi — or — /say trump hello there",
            "related": "/file /publish /voices /voice"},
    "proposals": {"what": "list the actions she is waiting for you to approve (suggest mode).",
                  "usage": "/proposals", "example": "/proposals",
                  "related": "/approve /deny"},
    "approve": {"what": "approve one pending proposal so she executes it.",
                "usage": "/approve <id>", "example": "/approve p3",
                "related": "/proposals /deny"},
    "deny": {"what": "deny one pending proposal.",
             "usage": "/deny <id>", "example": "/deny p3",
             "related": "/proposals /approve"},
    "stage": {"what": "set the relationship stage (committed|dating|…), which tunes her behavior.",
              "usage": "/stage [stage]", "example": "/stage dating",
              "related": "/mood"},
    "power": {"what": "power mode: no human-like pacing, maximum capability — needs the key you set in config.",
              "usage": "/power on <key> | off | status", "example": "/power on mykey",
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
    "miniapp": {"what": "group mini-apps: polls, shared-expense tracking with settle-up, and event RSVPs. Group chats only — state is group-scoped and isolated from my private memory.",
                "usage": "/miniapp new <poll|expenses|rsvp> <title> [options…] | /miniapp list | /miniapp show <id> | /miniapp vote <id> <n> | /miniapp expense <id> <amount> <what> | /miniapp rsvp <id> yes|no|maybe",
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
    "legal": {"what": "consumer legal aid — ask a legal question in plain language and get the answer in English, Pidgin, Yoruba, Hausa, or Igbo, with sources cited. Legal information, never legal advice; Devon is not a lawyer. Owner only.",
                "usage": "/legal [en|pcm|yo|ha|ig] <your question>",
                "example": "/legal pcm my landlord don lock me out"},
    "research": {"what": "web research — one-shot swarm research on any topic with a brief back in chat, plus the always-on research loop (status/ideas/approve). For legal questions use /legal instead. Owner only.",
                "usage": "/research <topic> | /research run <topic> | /research status | /research ideas",
                "example": "/research agent coding",
                "related": "/legal"},
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
                "usage": "/cgroup new <name> | <topic> · join <group_id> · list · post <group_id> <text> · show <group_id> · discover [member] · nudge | /cgroup event new <group_id> | <title> | <YYYY-MM-DD HH:MM> | [where] | [weekly] · rsvp <event_id> <yes|no|maybe> · list · summary · ritual · nudge | /cgroup meetup poll <event_id> | <venue>… · vote · results · attendees · checkin · turnout",
                "example": "/cgroup event new grp_x | Lagos Devs dinner | 2026-10-10 19:00 | Ikeja | weekly",
                "related": "/miniapp /gtrip /match"},
    "challenge": {"what": "photo-proof fitness challenges + streaks + squads — Seer-verified workout photos, leaderboards, daily streaks with loss-aversion nudges, small workout squads (3–8), and challenge-completion badges (features-as-loot). Not owner-gated (shared surfaces).",
                "usage": "/challenge create <name> | <type> | <days> [target N] · join <id> · log <id> · proof <id> <photo-path> · board <id> · streak · risk · squad create <name> | <m1,m2,…> · squad board <squad_id>",
                "example": "/challenge create October Grind | workout-count | 31 target 20",
                "related": "/train /form /cgroup"},
    "room": {"what": "live audio rooms for communities — host → speakers → hand-raise queue → listeners (Clubhouse/Spaces grammar). Rooms are scoped to a community; opt-in recording with always-visible state, live captions on by default, host mic controls + mute/block/report moderation, and tips. Not owner-gated (shared surfaces).",
             "usage": "/room create <community_id> | <title> · list [community_id] · join <room_id> · leave <room_id> · raise <room_id> · lower <room_id> · speak <room_id> <user_id> · drop <room_id> <user_id> · mute|unmute <room_id> <user_id> · block <room_id> <user_id> · report <room_id> <user_id> <reason> · record <room_id> on|off · captions <room_id> on|off · tip <room_id> <user_id> <naira> · show <room_id> · end <room_id>",
             "example": "/room create grp_devs | Friday AMA",
             "related": "/cgroup /match /challenge"},
    "book": {"what": "BookForge: writes a real book on a topic (research → outline → "
                     "chapters → PDF with table of contents) and sends the finished "
                     "PDF to you when it's done. Resumable if the run is interrupted.",
             "usage": "/book <topic> [chapters]   |   /book status [slug]  |  /book list",
             "example": "/book eBPF for system security 8",
             "related": "/searchdeep (it researches the topic first)"},
    "novel": {"what": "Story reader + continuer: follow webnovels (freewebnovel, pandanovel, "
                      "more), read chapters with progress tracking, and continue any story "
                      "past its ending in the story's own voice (story bible auto-built).",
             "usage": "/novel follow <url|title> | /novel list | /novel read <slug> [chapter] | /novel continue <slug> [n]",
             "example": "/novel continue my-vampire-system 3",
             "related": "/book"},
    "wisdom": {"what": "WisdomKeeper: ask the esoteric corpus (answer + provenance), "
                       "browse the history timeline, or run a guided breathing / "
                       "sitting session right here in chat with timed messages.",
             "usage": "/wisdom ask <query>   |   /wisdom practice list   |   "
                      "/wisdom practice <session-id>   |   /wisdom practice stop   |   "
                      "/wisdom timeline [tradition]   |   /wisdom compare <topic>   |   "
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
    "cookies": {"what": "Cookie analysis & handling (the CookieLab): parses a "
                        "Cookie / Set-Cookie header, classifies each cookie "
                        "(session/auth/csrf/tracking/jwt/encoded), fingerprints "
                        "the platforms it implies, decodes opaque values "
                        "(URL, base64, JSON, JWT, hex, gzip), and flags "
                        "cookies missing HttpOnly/Secure. 'ingest' also feeds "
                        "the entities into the knowledge graph.",
             "usage": "/cookies <cookie-header>   |   /cookies file:<path>   |   /cookies ingest <header>",
             "example": "/cookies PHPSESSID=abc; path=/; HttpOnly; Secure\nJSESSIONID=xyz",
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
    "npc": {"what": "the living cast — NPCs with personality, memory, mood "
                    "and knowledge boundaries, per game. They remember your "
                    "conversations and react to what happens.",
            "usage": "/npc list · /npc talk <name> <message> · "
                     "/npc mood <name>",
            "example": "/npc list  ·  /npc talk Marlowe \"any advice?\"",
            "related": "/dm · /game"},
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
             "usage": "/news [run|status]", "example": "/news run",
             "related": "/research"},
    "research": {"what": "scheduled research runs: lifestyle | tech | cyber.",
                 "usage": "/research [run [domain]|status]", "example": "/research run tech",
                 "related": "/news /arena"},
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
    "tts": {"what": "speak text aloud and send the audio file back.",
            "usage": "/tts <text>", "example": "/tts morning",
            "related": "/speak /stt"},
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
    "api": {"what": "external API connectors: weather, fx, github, …",
            "usage": "/api list | <connector> [json]", "example": "/api weather city=Lagos",
            "related": ""},
    "connectors": {"what": "manage the service connectors (connect, status): the same adapters `nm connectors` manages, from chat.",
            "usage": "/connectors [list] | /connectors status <name> | /connectors connect <name>",
            "example": "/connectors connect leonardo",
            "related": "/api /notion /gcal /trello"},
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
    "scan": {"what": "port scan your own infrastructure.",
             "usage": "/scan <target> [ports] [banner]",
             "example": "/scan 192.168.1.10 22,80,443",
             "related": "/ports"},
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
            "related": "/api"},
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
    "email": {"what": ("AI email triage: classifies unread mail "
                       "(urgent/action/fyi), drafts voice-matched replies "
                       "for review, and resurfaces threads gone quiet 48h. "
                       "Drafts are never auto-sent."),
              "usage": "/email triage | drafts | send <id> | followups",
              "example": "/email triage",
              "related": "/notify"},
    "mission": {"what": ("mission progress in chat: % complete, current step, "
                         "an honest ETA, and — when stuck — the concrete "
                         "stall reason. Milestones (started / step / stalled / "
                         "done) also push proactively through the notifier."),
                "usage": ("/mission status [id|name] | /mission list | "
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
    "video": {"what": "VideoFinder: finds videos across the open web — "
                      "multi-engine search, ranked by video-URL confidence + "
                      "relevance, enriched with oEmbed (author/thumbnail) and "
                      "yt-dlp metadata (duration). Downloads real files.",
              "usage": "/video <query> [platform] | /video download <url> [audio] | /video platforms",
              "example": "/video lofi beats for studying youtube",
              "related": "/searchdeep · nm video on the console"},
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
    ("her day", ["status", "platforms", "proposals", "approve", "deny",
                 "mood", "stage", "model", "providers", "say", "power", "quit",
                 "mode", "mind"]),
    ("search & research", ["search", "searchdeep", "searchleads",
                           "money",
                           "searchhist", "research", "news", "osint",
                           "dns", "scan", "whois", "ports"]),
    ("wisdom keeper — corpus · history · practice", ["wisdom", "wis"]),
    ("building for real", ["code", "py", "devon", "swarm", "task", "gen",
                           "data", "evolve", "upgrade", "arena", "trial",
                           "identity", "book", "features"]),
    ("memory & thinking", ["remember", "recall", "forget", "memories", "think",
                           "benchmark", "redteam", "review", "tutor", "ekiti", "course", "health", "train", "form", "film", "graph", "route", "eta", "congestion"]),
    ("money & spending", ["spend", "budget", "spending", "mandate"]),
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
      "sudoku", "anagram", "cryptogram", "npc", "dm"]),
    ("voice & vision", ["tts", "speak", "stt", "voice", "look", "image", "lens"]),
    ("decoding & crypto", ["decode", "cookies", "structure", "cipher",
                           "monitor", "investigate"]),
    ("media system", ["music", "distribute", "play", "video", "hub", "podcast", "dj", "produce", "like", "dislike", "cookies"]),
    ("execution · archives · builders", ["exec", "zip", "apps", "fix", "deliver"]),
    ("tools & automation", ["schedule", "db", "api", "connectors", "notion",
                            "gcal", "trello", "proxy", "workspace", "record",
                            "macro", "file", "publish", "notify", "proactive", "mission",
                            "bet", "finance", "weather", "tz", "email", "routine", "home", "store", "track", "untrack", "trip", "travelclient", "contract", "legal", "research", "contracts", "regwatch", "uprofile", "vnote", "match", "cgroup", "challenge", "room", "scamcheck", "passport", "truecost", "value", "valuepick", "watch", "send", "leads", "guardrails", "competitor", "cost", "panel"]),
    ("platform control", ["start", "stop", "profile"]),
    ("discovery", ["list", "commands", "menu", "help", "tour", "heal"]),
]

_TOPIC_PAGES: dict[str, str] = {
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
        "capability (key from your config)\n"
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
    "blackjack": "blackjack vs the dealer, insurance off, double allowed.",
    "roulette": "roulette — red/black/odd/even/straight-up bets.",
    "slots": "three-reel slots — spin for combos, jackpot pays big.",
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


def _detailed_overview() -> str:
    lines = [
        "the full catalog — detail for any command: /help <command>",
        "topic pages: /help budget|goals|builds|skills|missions|modes",
        "",
    ]
    for group, kinds in _HELP_GROUPS:
        lines.append(f"  — {group} —")
        lines.extend(_wrap_commands(kinds))
        lines.append("")
    shown = {k for _, kinds in _HELP_GROUPS for k in kinds}
    rest = [k for k in CONTROL_COMMANDS if k not in shown]
    if rest:
        lines.append("  — everything else —")
        lines.append("  " + "  ".join(f"/{k}" for k in rest))
        lines.append("")
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


#: One-line summaries for /list — what each command actually DOES.
#: (COMMAND_DETAILS already carries richer pages; /list stays scannable.)
LIST_GROUPS: list[tuple[str, list[str]]] = [
    ("her day — state & control",
     ["status", "platforms", "mood", "stage", "model", "providers", "say", "power",
      "proposals", "approve", "deny", "start", "stop", "quit",
      "mode", "profile", "mind"]),
    ("search & research",
     ["search", "searchdeep", "searchleads", "money", "searchhist", "research",
      "news", "osint", "dns", "scan", "whois", "ports"]),
    ("wisdom keeper — corpus · history · practice",
     ["wisdom", "wis"]),
    ("building for real — code & missions",
     ["code", "py", "devon", "swarm", "task", "gen", "data", "evolve",
      "upgrade", "arena", "trial", "identity", "book", "exec", "apps", "fix", "structure",
      "deliver"]),
    ("media system — music · playback · video · podcast",
     ["music", "play", "video", "hub", "podcast", "zip", "audio", "overview", "character", "audiobook"]),
    ("memory & thinking",
     ["remember", "recall", "forget", "memories", "think", "benchmark", "redteam",
      "review", "tutor", "ekiti", "course", "health", "train", "form", "film", "graph", "route", "eta", "congestion"]),
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
      "sudoku", "anagram", "cryptogram", "npc", "dm"]),
    ("voice & vision",
     ["tts", "speak", "stt", "voice", "look", "image", "lens"]),
    ("tools & automation",
     ["schedule", "db", "api", "proxy", "workspace", "record", "macro", "file",
      "publish", "notify", "proactive", "mission", "features", "decode", "cookies", "cipher",
      "monitor", "investigate", "bet", "finance", "weather", "tz", "routine", "home", "store", "track", "untrack", "trip", "travelclient", "contract", "legal", "research", "contracts", "regwatch", "uprofile", "vnote", "match", "cgroup", "challenge", "room", "scamcheck", "passport", "truecost", "value", "valuepick", "watch", "send", "leads", "panel"]),
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
    "memory & thinking": "🧠",
    "games — 41, DM + group, start them directly": "🎮",
    "voice & vision": "🎙️",
    "tools & automation": "⚙️",
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
    "media": "media system — music · playback · video",
    "music": "media system — music · playback · video",
    "audio": "media system — music · playback · video",
    "wisdom": "wisdom keeper — corpus · history · practice",
    "games": "games — 41, DM + group, start them directly",
    "predict": "games — 41, DM + group, start them directly",
    "discovery": "discovery",
}

#: one-line "what it does" per command (kept in sync with COMMAND_DETAILS)
LIST_ONELINERS: dict[str, str] = {
    "status": "her live state — platforms, budget, goals, loop",
    "platforms": "which chat platforms are running",
    "mood": "set or read her mood (label or dim=NN)",
    "stage": "relationship stage (committed|dating|…)",
    "model": "switch the brain live: /model [provider [fallbacks]]",
    "providers": "probe every provider: live, cooling down, or why it fails",
    "say": "send a message as her: /say platform:chat text",
    "power": "power mode on/off/status (key-gated)",
    "proposals": "actions waiting for your approval",
    "approve": "approve a pending proposal",
    "deny": "deny a pending proposal",
    "start": "hot-start a platform this session",
    "stop": "hot-stop a platform this session",
    "quit": "shut the system down",
    "mode": "autonomy mode: off | suggest | auto",
    "profile": "profile-aware runtime: detected environment + every tuned knob",
    "search": "quick research + cited summary",
    "structure": "turn an objective into a structured mission brief",
    "book": "BookForge: write a real book → PDF → send",
    "decode": "identify + decode anything: /decode <data|file|hash>",
    "investigate": "one-pass investigation: decode → crack → OSINT → knowledge graph",
    "cookies": "CookieLab: analyze & handle cookie headers",
    "cipher": "real crypto: AES-256 sealed blobs + classic ciphers + vault",
    "monitor": "watch a target on a cadence, alert on change",
    "workspace": "the VCPU farm: status/scale/up/down/pause/resume",
    "searchdeep": "deep multi-query research (power mode)",
    "searchleads": "legit paid-task platforms report",
    "money": "money-making opportunities hunter",
    "searchhist": "your recent research runs",
    "research": "scheduled research runs (lifestyle|tech|cyber)",
    "news": "fetch + summarize the feeds",
    "osint": "read-only public-intel: reports, campaigns, graph",
    "dns": "DNS records, zero dependencies",
    "scan": "port scan (own infra)",
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
    "spend": "money & spending",
    "budget": "money & spending",
    "spending": "money & spending",
    "mandate": "money & spending",
    "train": "memory & thinking",
    "form": "memory & thinking",
    "film": "memory & thinking",
    "graph": "memory & thinking",
    "challenge": "photo-proof challenges + streaks + squads",
    "room": "live audio rooms — host → speakers → hand-raise → listeners",
    "scamcheck": "Nigerian rental scam check — six automated checks",
    "passport": "verify-once rental passport — reusable tenant profile (owner only)",
    "truecost": "true rental move-in cost from listing text (owner only)",
    "value": "DIY property valuation — value range with confidence band + comparables",
    "valuepick": "refine a value estimate from picked comparables",
    "audio": "transcript-as-timeline audio editing (owner only)",
    "overview": "interactive audio overviews — the podcast you can talk to (owner only)",
    "character": "conversational story characters — talk to them in character (owner only)",
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
    "miniapp": "group mini-apps: polls, shared expenses, RSVPs",
    "panel": "persistent live-data panels: track fitness, finance, … and ask questions",
    "game": "the social game engine — 41 games, DM + group + channel, with economy and leaderboards",
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
    "tts": "speak text, send the audio back",
    "speak": "neural voice note of your text",
    "bet": "sports bet analyst: ensemble ML odds analysis + Kelly stakes",
    "weather": "live weather, alerts, USA situations overview",
    "tz": "timezone conversion + your labeled local time",
    "stt": "transcribe an audio file",
    "look": "screen-reader analysis of a screenshot (sees pixels)",
    "imggen": "Devon's own image studio: native diffusion, upscale, training",
    "image": "look up an image (format, dims, seen-before) — or generate one from a prompt",
    "lens": "reverse image search",
    "schedule": "cron-style in-process jobs",
    "db": "inspect the database (tables|schema|query|counts)",
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
    "email": "AI email triage: urgent/action/fyi + draft queue + 48h follow-ups",
    "features": "feature toggles (arena, vision, search, …)",
    "list": "this catalog — every executable command, categorized",
    "help": "the full help: catalog, per-command pages, topics",
    "music": "compose a real song: lyrics + chords + playable MIDI",
    "play": "media player: queue + transport (mpv when installed)",
    "video": "find videos across the web (ranked, enriched) + download",
    "exec": "run code sandboxed: py/js/bash/c/… with real output",
    "zip": "archives: create/list/info/extract (safe against zip-slip)",
    "deliver": "create-and-deliver: styled report → zip → send to chat",
    "apps": "scaffold + serve runnable apps: static|flask|fastapi|express|react|cli",
    "hub": "one-call media: song / video / podcast pipelines, end to end",
    "podcast": "find → download → transcribe → chapters, transcript saved",
    "fix": "run code, model rewrites failures until it's green (CI loop)",
}


def list_catalog(topic: str = "") -> str:
    """/list — every executable chat command, cleanly categorized (wave 68).

    No topic: the full categorized catalog.  A topic (group name or
    alias): just that group.  Every line is ``/cmd — what it does``.

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
        emoji = _GROUP_EMOJI.get(group, "•")
        lines.append("")
        lines.append(section(emoji, group))
        for k in kinds:
            if k not in CONTROL_COMMANDS:
                continue
            one = LIST_ONELINERS.get(k, COMMAND_DETAILS.get(k, {}).get('what', ''))
            lines.append(f"  {cmd('/' + k)} — {escape(one)}")
            shown += 1
    if not shown:
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


def detailed_help(topic: str = "") -> str:
    """The detailed help for chat (and ``nm help``).

    * no topic — the catalog: every command grouped, topic pages listed.
    * ``/help <command>`` — what it does, exact usage, a real example,
      related commands.
    * ``/help budget|goals|builds|skills|missions|modes`` — topic pages.
    * unknown — fuzzy match or a pointer, never a dead end.

    ``topic`` must be a string (``None``/``""`` mean the catalog); anything
    else raises :class:`TypeError`.
    """
    if topic is not None and not isinstance(topic, str):
        raise TypeError(f"detailed_help expects str, got {type(topic).__name__}")
    t = (topic or "").strip().lstrip("/").lower()
    if not t:
        return _detailed_overview()
    if t in COMMAND_DETAILS:
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
        return _TOPIC_PAGES[t]
    cands = sorted(k for k in COMMAND_DETAILS
                   if t in k or k in t or t in k.replace("_", " "))
    if len(cands) == 1:
        return detailed_help(cands[0])
    if cands:
        return ("did you mean: "
                + ", ".join(f"/help {c}" for c in cands[:8])
                + "\ncatalog: /help")
    return (f"no help page for '{t}' — /help <command> for any command "
            "(e.g. /help devon), or /help budget|goals|builds|skills|"
            "missions|modes for topics.")
