# Personal bot — BotFather command list (USER-ONLY)

The personal bot is **owner-only**: only the owner can use it, so its
BotFather `/setcommands` list shows **everything** — the public 52
commands PLUS the 48 owner-only commands = **100 total** (the BotFather
maximum). The gated bot's public list stays exactly as-is (52 commands,
owner-only hidden) — this file does not change it.

## How to set it in BotFather

1. Open **@BotFather** in Telegram.
2. Send `/setcommands`.
3. Pick the **personal** bot (NOT the gated/public bot).
4. Paste the whole block below (100 lines), exactly as-is.

BotFather format: one `command - description` per line, command WITHOUT
the leading slash, description at most 256 chars.

## The list (generated — do not hand-edit)

The block below is generated from the command registry
(`nomorals/social/chat/personal_commands.py`) — descriptions come from
`COMMAND_DETAILS`. A test (`tests/test_personal_bot.py`) asserts this
doc and the generator agree, so hand edits here would fail the suite:
edit the registry / generator, re-render, and paste the fresh block.

```
help - this help — the full catalog, or the detail page for one command.
search - quick research: searches and replies with a cited summary.
news - fetch and summarize the news feeds.
book - BookForge: writes a real book on a topic (research → outline → chapters → PDF with table of contents) and sends the finished PDF to you when it's done.
music - MusicCreator: composes a REAL song from a topic — style-aware lyrics, section structure, chord progression, melody description, and a playable .mid file (real MIDI, opens in any player).
play - media player: durable queue + transport for audio.
video - VideoFinder: finds videos across the open web — multi-engine search, ranked by video-URL confidence + relevance, enriched with oEmbed (author/thumbnail) and yt-dlp metadata (duration).
podcast - The podcast pipeline, one call: find a video/podcast for the query, download the audio, transcribe it (OpenAI-compatible STT or whisper.cpp when available), summarize, split into titled chapters, and save the transcript.
tts - speak text aloud and send the audio file back.
stt - transcribe an audio file to text.
look - screen-reader analysis of a screenshot or image URL (she actually sees the pixels).
image - look up an image (path/URL) or generate one from a text prompt.
lens - reverse image search.
decode - Universal Decoder: identifies and decodes almost anything — base64/hex/base32/base58/binary, ROT13, leetspeak, URL, hashes (with known-secret match), cookies, JWTs, tokens, and raw file forensics.
dns - DNS records with zero dependencies: A/AAAA/MX/NS/TXT/SPF/CAA.
whois - domain registration data via RDAP.
cipher - Real cryptography: AES-256 (PBKDF2) sealed blobs with an integrity tag — wrong passphrases and tampered blobs are rejected, not returned as garbage.
wisdom - WisdomKeeper: ask the esoteric corpus (answer + provenance), browse the history timeline, or run a guided breathing / sitting session right here in chat with timed messages.
think - explicit multi-step reasoning with the full trace shown.
game - the social game engine — 41 games across DM, group and channel, with a shared economy, items and leaderboards.
wordchain - word chain — each word must start with the last letter of the previous one; the house breaks dead ends.
hangman - hangman — the house picks a word, you send letters one at a time; the board draws as you miss.
numberguess - number guess — the house thinks of a number, you narrow the range.
two_truths - two truths and a lie — everyone posts three claims, the table votes for the lie.
wyrr - wyrr — the house poses, the table picks, everyone sees who's who.
spy - spy — everyone gets a word but one of you gets a fake; find the spy.
auction - auction — the table bids its points up for what the house puts on the block.
trivia - trivia royale — timed trivia rounds, the table races the clock and each other.
20q - 20 questions — she thinks of something, you ask yes/no (20 max).
rps - rock paper scissors — best of five vs the house.
digits - digit memory — repeat the digits back, they grow every round.
mafia - mafia — town vs mafia with a night phase; the house runs the lynch and the kills.
king - king of the hill — climb the hill and defend your spot from challengers.
story - story chain — one sentence each, the story grows, the house judges the arc.
rpg - rpg adventure — a shared dungeon crawl: the house sets the scene, the table decides.
shop - shop game — the in-chat market turns into a game of spending and sniping.
duel - quiz duel — one-on-one rapid-fire questions, best of the round wins.
case - the case — a cooperative investigation: clues drop, the table reasons together.
world - world — a continuing town that never ends, it just grows.
escape - escape room — 4 locks, 3 strikes each, the table breaks out together.
political - political — 3 elections: pledge, campaign, vote, be mayor (groups).
poker - heads-up limit poker vs the table (Devon deals).
ttt - tic-tac-toe, 3×3, best of five.
bulls - bulls & cows — guess the 4-digit number, clues per try.
craps - craps table — pass/don't't pass, come-out rolls and points.
memory - memory match — flip pairs, fewest moves wins.
mines - minesweeper cashout — step on a mine and the pot is gone.
wordle - wordle — five letters, six guesses, shared daily word.
2048 - 2048 — slide and merge to 2048 on a 4×4 board.
snake - snake — eat, grow, don't bite yourself.
connect4 - connect four — drop discs, line four up.
battleship - battleship — your fleet vs hers, 10×10.
status - her live state — platforms, model budget, active goals, loop cadence.
platforms - which chat platforms are running in this session.
mood - set her mood — a label (happy, tired, playful, calm, …), or set dimensions directly (0-100): affection, happiness, energy, trust, jealousy, frustration, intimacy, distance, insecurity, pride.
stage - set the relationship stage (committed|dating|…), which tunes her behavior.
model - switch the active model live (up to 5 fallbacks); no argument shows the chain.
providers - probe every registered provider and report live/dead with the reason — the diagnostic for a silent brain.
say - rich voice: emotion, accent, ambience, room reverb.
power - power mode: no human-like pacing, maximum capability — needs the key you set in config, or the owner seal (identity + passphrase).
proposals - list everything waiting for your approval: proactive actions + research upgrade proposals.
approve - approve one pending proposal so she executes it (upgrade tickets are approved + test-gated applied).
deny - deny one pending proposal (upgrade tickets need a reason).
start - hot-start a chat platform without restarting the system.
stop - hot-stop a chat platform (it keeps running data on disk).
quit - shut the whole system down (everything is already on disk).
mode - her autonomy level: off (asleep) | suggest (proposes, you approve) | auto (acts).
profile - the profile-aware runtime: detected environment (termux/mobile/pc/vps/workstation) and every tuned knob — threads, VCPUs, download caps, memory pressure, parallel chats, mission aggressiveness, model preference — with the source of each value.
mind - the Core Mind — the always-on layer that turns a natural-language goal into routed work: research swarm, builder, browser, downloader, missions, games, orchestrator.
features - list or toggle system features: arena, group_posts, proactive_dm, vision, search.
email - the owner's mailbox over Gmail: list/search/read mail, classify unread mail (urgent/action/fyi), draft voice-matched replies for review, and resurface threads gone quiet 48h.
update - Self-update from chat: mission gate (refuses mid-mission), checkpoint resumable missions, transactional pull + migrate + health-check (auto-rollback on failure), then restart and resume.
pull - Pull the latest code (git pull --ff-only) without restarting.
deploy - Deploy this code: same as /update (gate + checkpoint + pull + restart).
restart - Restart the process: mission gate, checkpoint resumable missions, restart, resume from checkpoints.
searchdeep - deep multi-query research with synthesis (power mode).
money - hunts every kind of money-making opportunity — paid tasks, referrals, free courses, bounties, gigs — ranked by value vs effort.
searchhist - your recent research runs.
research - scheduled research runs: lifestyle | tech | cyber.
code - the coding bot: drafts code, runs it in the sandbox, fixes until the acceptance command passes; the real file lands in her workspace.
py - run python in the sandbox; -s name keeps a session, -r resets it.
devon - the autonomous dev & investigation agent: he plans the tools himself, runs them, digests, replies — for anything you say.
task - queued instructions she runs and reports back on.
mission - mission progress in chat: % complete, current step, an honest ETA, and — when stuck — the concrete stall reason.
schedule - cron-style jobs that run in-process: at / every / daily.
monitor - Watches a URL or file and alerts you (with a short diff) when its content actually changes.
immune - defensive immune system — thymus negative selection, innate scan, antibody memory, seven-rung revoke-only police, fever, drift, case files.
impossible - Impossible Made Possible — always-on cross-domain conceive-and-solve: formulate, classify, prior art, TRIZ transfer, feasibility gate, falsifier, attempt.
arena - the content arena: research a topic, stream drafts, review, export, approve builds.
trial - plan / store / send ONE trial-account signup you asked for (stored encrypted, one account); full auto signups (forms, CAPTCHA solver, temp email/SMS) with a confirmation gate.
exec - Execution system: runs code in the sandbox with structured results (stdout, stderr, exit code, wall time, timeout flag, files written).
db - inspect the database: tables, schema, counts, or a query.
remember - store something in her long-term memory.
recall - what she remembers, ranked; a query narrows it.
forget - delete a memory by id or description.
memories - trust view: how many people she tracks, how many long-term memories she holds, recent contact.
spend - log a spend in Naira.
budget - monthly spending budgets per category.
spending - spending summary vs your budgets, with a weekly digest view.
list - every command you can run from chat, cleanly categorized — this catalog.
```

## Inline buttons (personal bot only)

Every personal-bot reply carries an inline keyboard with relevant action
buttons + next actions, when the origin command has sensible ones:

| Reply to | Buttons |
|---|---|
| `/help` / `/list` | Category jumps: Search, Media, Games, Power, Build, Memory, Voice, Wisdom, Tools, Control, All commands (each runs `/list <group>`, Power runs `/power status`) |
| `/power` | Power On, Power Off, Power Status |
| `/search` | Search again, Deep research, Save query, Commands |
| `/searchdeep` | Search again (retry), Commands |
| `/task` | Task list, Commands |
| `/mission` | Mission status, Mission list |
| `/status` | Power status, Providers |
| `/providers` / `/model` | Status/Providers, Commands |
| `/impossible` | Queue, Examples |
| `/book` | Book list, Book status |
| game results | Rematch, Next, Arena, Expert (shared pattern) |

Rules: every button routes to a real command (no dead buttons); a button
whose payload would exceed Telegram's 47-byte callback budget is omitted,
never sent truncated; **Power On carries NO key** — the owner key must
never travel in a callback, so tapping it routes to the /power handler,
which replies with usage; replies with no sensible next action get no
keyboard. The gated bot's replies are unchanged (text + its existing
shared patterns only).

