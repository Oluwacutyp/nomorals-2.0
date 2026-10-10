# GAMES Sweep — External Mining Report

Mined 2026-10-10 for the system-wide upgrade of `nomorals/games/` (55 files).
Every significant class compared against the best implementation outside the repo —
best-in-class AND the trash. Gold below, merged into the real classes (no parallel systems).

## 1. GameEngine (engine.py) — chat room game orchestration

**How the best do it:** Lichess runs authoritative servers: every move validated
server-side, state serialized after each turn, timeouts handled by a sweeper, not by
client goodwill. Discord game bots (the serious ones) keep a room object per channel,
a background task for turn timers, and never trust the client. Research sources agree:
server-driven turn management, authoritative state, graceful disconnect handling,
visible turn indicators/timers, feedback on invalid moves.
Sources: fedbyai/staging.canadagamescentre turn-based multiplayer best-practice guides.

**Gaps found:** no spectator concept, no rematch flow, game-over presentation is plain.
**Gold taken:** rich game-over card (stats, payout breakdown, rematch offer),
post-game summary lines. Not taken: client prediction (irrelevant for chat).

## 2. Matchmaking / ratings (matchmaking.py) — Elo only

**How the best do it:** OpenSkill.py (Weng & Lin, JMLR 2011 — license-free TrueSkill
alternative): per-player (μ, σ) with conservative ordinal μ−3σ used for matchmaking
so uncertain newcomers don't get lopsided matches; Plackett-Luce model scales to
N-player fields. pithy-sh rating docs: separate **skill rating (MMR, hidden, moves
up/down)** from **experience (XP, visible, monotonic)** — one number can't do both jobs.
Glicko-2 adds rating deviation that grows with inactivity (rust) and volatility.
Sources: github.com/vivekjoshy/openskill.py, wdm0006/elote docs, pithy-sh/pithy
`packages/rating/docs/algorithms.md`.

**Gaps found:** flat Elo, no uncertainty, no inactivity decay, no placement matches,
no tiers, no win-probability display, queue pairs on raw Elo.
**Gold taken:** full Glicko-2 (μ/σ/φ persisted per game, RD inflation on read by
inactivity days), conservative ordinal μ−3σ for queue pairing, placement matches
(first 5: gains ×2, losses don't drop below start), rank tiers
(Bronze→Mythic), win-probability prediction for fair-pairing and upset detection.
Backwards compatible: `record_elo`/`get_rating` untouched.

## 3. GameMind (ai.py) — connect4 alpha-beta, battleship hunt/target, reversi

**How the best do it:** MCTS (AlphaGo's core): game-agnostic UCT search with a time
budget — scales with thinking time, works on any game implementing legal_moves/simulate.
For battleship, best-in-class bots use a **probability density heatmap**: enumerate every
placement consistent with known hits/misses, count how often each cell is occupied,
shoot the max — strictly stronger than hunt/target parity heuristics.
Connect4: solver-quality via alpha-beta with transposition tables; MCTS as the scalable
anytime alternative.
Sources: afreefaw/monte-carlo-tree-search (game-agnostic MCTS), towardsdatascience
Monte-Carlo connect4, carloconnect thesis (MCTS beats flat Monte-Carlo at equal budget).

**Gaps found:** no anytime search, no difficulty tiers (house is one strength — bad for
retention), battleship AI has no probability reasoning.
**Gold taken:** game-agnostic `MCTSEngine` (selection/expansion/simulation/backprop,
time-budgeted, UCT), `connect4_move(..., difficulty=)` — easy (greedy+noise),
medium (alpha-beta), hard (MCTS); battleship probability heatmap mode that enumerates
consistent placements.

## 4. Achievements (achievements.py)

**How the best do it:** Xbox (2016+): every achievement carries a live unlock %
— <10% = "Rare" with distinct sound/animation; <5% even rarer. Steam: same rarity
percentages + player-curated profile showcases. Player research asks for: repeatable
achievements (re-earn "1000 kills" at 2000), Gamerscore that floats with rarity,
"celebrate past achievements" anniversaries, progress-tracked multi-step achievements,
and never confusing completion with mastery.
Sources: rtfenter/player-accomplishment-product-study, windowscentral Xbox rarity
explainer, resetera/Xbox community asks.

**Gaps found:** static rarity strings (no live % basis), no progress display on
count-based achievements, no showcase, no repeatables, no anniversaries.
**Gold taken:** live rarity engine (unlock % → common/uncommon/rare/epic/legendary,
Xbox cutoffs), tracked achievements with progress (`track_progress`, render bars),
repeatable milestones (re-earn every N), achievement anniversaries
("unlocked this day 1 year ago"), player showcase (pin 3).

## 5. GameMaster (gamemaster.py) — LLM DM with template fallback

**How the best do it:** tsueagnes/nlp_2-agentic_rpg_gamemaster: supervisor + specialist
components (world lore, NPC behavior, quest progression, rules resolution, continuity
checking, state updates, narration) — structured data kept OUT of prompts, continuity
checked as its own step. staphit/AI-Dungeon-Master: scripted-branch mode vs free mode,
SQLite campaign persistence, exportable story. RealmTorch: quests/NPCs/locations/factions
as first-class managed objects, random generators for everything.
Sources: github.com/tsueagnes/nlp_2-agentic_rpg_gamemaster,
github.com/staphit/AI-Dungeon-Master, enworld RealmTorch thread.

**Gaps found:** narration is event-driven but quests/NPCs are ephemeral; no quest
objects with objectives/twists/rewards; no scene framing; no continuity log.
**Gold taken:** structured `Quest` objects (objective, beats, twist, reward, expiry),
`SceneCard` framing (location/mood/stakes/cliffhanger), persistent quest log per game,
continuity notes appended to prompts. Template-forge fallbacks kept.

## 6. Tournaments (tournaments.py) — points leagues only

**How the best do it:** Swiss (py4swiss/bbpPairings): score-group pairing, no rematches,
byes for odd fields, tiebreaks — Buchholz (strength of schedule), Sonneborn-Berger,
median-Buchholz. ryanwc/SwissTournament: wins → SOS → points scored → registration
order. Chess managers: Dutch pairing, color balancing, configurable scoring.
Single/double elimination: bracket generation with seeding; round robin: circle method.
Sources: github.com/moritz72/py4swiss, github.com/ryanwc/SwissTournament,
github.com/raphaelbabilonia/chess-tournament-manager.

**Gaps found:** only round-robin-points; no elimination brackets, no Swiss, no
tiebreaks (ties broken arbitrarily), no byes.
**Gold taken:** `swiss_pairings` (score groups, rematch avoidance, bye handling),
`double_elim_bracket` (winners/losers brackets, seeding), `round_robin_schedule`
(circle method), Buchholz + Sonneborn-Berger tiebreaks, `Tournament.format` field
("points" | "swiss" | "elimination" | "round_robin") with format-aware standings.

## 7. Gear (gear.py) — typed gear with grades/sets/durability

**How the best do it:** Diablo II item generation: magic items roll prefix/suffix
(25% both, 25% prefix-only, 50% suffix-only); rares get 3–6 affixes, ≤3 prefixes and
≤3 suffixes, **one affix per group max** (no double mana-group rolls), ilvl gates
which affix tiers can appear. Affix names build the item name ("Bronze Sword of the
Whale"). Last Epoch discussion: affixes gated by item class, not by build.
Sources: diablo2.diablowiki.net Item Generation Tutorial, diablo.fandom.com Affix,
brinven/dungeons-of-moria-reforged.

**Gaps found:** gear stats are fixed per blueprint — no randomized loot, no affixes,
no "one more drop" chase loop.
**Gold taken:** full affix system — `AFFIXES` with prefix/suffix groups (one-per-group
rule), ilvl-gated value tiers, affix count by grade (magic 1–2, rare 3–6),
`forge_loot(base, ilvl)` generating named items, affix bonuses folded into
`effective_stats`.

## 8. Economy (economy.py) — shop + wallet + rewards

**How the best do it:** F2P economy design (gamedeveloper.com "cake recipe"): fix
currency values to USD, enumerate **every source and every sink**, set $/day earn
rates per tier, balance so daily earnings < cheapest meaningful purchase. Two
currencies: soft (grind) + hard (premium/scarce). Sinks must scale faster than
benefits (repair fees, taxes, prestige cosmetics for whales). Battle passes: season
track with XP curve, daily caps, streak rewards, completion-refunds (Fortnite).
Sources: gamedeveloper.com F2P economy guide, senglaypann/indie-studio monetization
SKILL.md, huntervault battle-pass analysis, tentonhammer MMO sinks.

**Gaps found:** no source/sink ledger (can't audit inflation), no hard currency,
no price dynamism, no economy health signal.
**Gold taken:** `game_ledger` table (every coin movement tagged source/sink +
category), `economy_health()` faucet/drain ratio with verdict, gems (hard currency,
earned from achievements/seasons only — never bought), prestige gem items,
dynamic pricing (prices drift with server-wide earn velocity).

## 9. TriviaForge (trivia_forge.py) — LLM question generation

**How the best do it:** Millionaire-style ladder (escalating difficulty + prize),
lifelines (50:50, phone-a-friend/audience, skip), streak bonuses, fastest-finger
tiebreaks. OpenTDB model: difficulty-tagged banks, category pools.
**Gaps found:** questions dealt but no match structure — no ladder, no lifelines,
no stakes.
**Gold taken:** `TriviaLadder` match: escalating difficulty rounds, three lifelines
(50:50, skip, audience poll), streak multiplier, guaranteed-minimum walkaway,
rendered question cards.

## 10. Daily (daily.py) — one arena hunt/day

**How the best do it:** Duolingo: streaks are THE retention mechanic — streak count,
streak freeze (protects one missed day), streak repair (pay to restore), milestone
celebrations (7/30/100/365), calendar view, pushy but warm copy.
**Gaps found:** binary done/not-done, no streak, no freeze, no milestones.
**Gold taken:** streak tracking, streak freeze item (shop), streak repair for coins,
milestone celebrations, week-calendar render.

## 11. Seasons (seasons.py) — weekly events (kept as-is; battle-pass track added)

Fortnite model: pass pays back its cost on completion → rolling retention. Taken:
season XP track (free track tiers with rewards) computed from existing event
multipliers. (Light touch — seasons.py already solid.)

## 12. RPG spine (combat/stats/skills/enemies/npc/progression/mastery/titles/power/invent/lexicon)

Already deep (30K-line spine). Mining confirmed the shape matches best practice
(stat blocks, skill trees, enemy AI). No structural changes; affix system (7) and
achievements (4) feed it.

## Presentation style (all modules)

Best game UIs: consistent card headers, progress bars, tier badges, one-line
verdicts. Applied: `render_*` functions use shared bar/badge conventions from
`whatsapp_format.py` (`wa_bar`), tier emoji (🟤⚪🟡🟣🟠), rarity colors in text.
