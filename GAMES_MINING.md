# Games Phase Mining Report — 2026-10-10

## What exists (surveyed)

### Game library (`nomorals/games/games/`)
| File | Games |
|---|---|
| `easy.py` | WordChain, Hangman, NumberGuess, TwoTruths, Wyrr, Spy, Auction, **TriviaRoyale**, TwentyQuestions, RPS, DigitMemory |
| `medium.py` | Mafia, KingOfHill, StoryChain, RPGAdventure, Shop, QuizDuel, Investigation |
| `wild.py` | Poker, TicTacToe, BullsCows, Craps, Concentration, Minesweeper, Wordle |
| `arcade.py` | 2048, Snake, ConnectFour, Battleship |
| `casino.py` | Blackjack, Roulette, Slots |
| `puzzles.py` | Sudoku, Anagram, Cryptogram |
| `pvp.py` | Duel, Raid (arena combat) |
| `ambitious.py` | WorldGame, BattleArena, EscapeRoom, PoliticalGame |
| `inbox.py` | Gomoku, Reversi, Checkers |
| `case_bank/` | 52 hand-written cases + template `generate_case()` (seeded, solvable) |

### Infrastructure
- `engine.py` — platform-agnostic rooms, turns, timers, persistence, scheduler
- `gamemaster.py` — DM narration with moods, NPC cast, LLM quest/item forge w/ template fallback
- `players.py` — Player store, leaderboards, AI seats
- `economy.py` — coins, shop, XP
- `achievements.py`, `mastery.py`, `titles.py`, `seasons.py`, `tournaments.py`
- `fairness.py` — anti-cheat
- `relay.py` — cross-platform?

### Arena (`nomorals/agents/arena/`)
- `topics.py` — 170 topics, 14 categories, interest-profile sampling, anti-repeat window (persisted), topic packs registry
- `activity.py` — builds interest profile from user behavior
- `sampling.py`, `challenges.py`, `scoring.py`

### Already dynamic (good)
- Arena topics: personalized + anti-repeat + packs ✓
- Case game: `generate_case()` seeded template composer, `deal_case` anti-repeat, `adapt_case` difficulty tuning ✓
- GameMaster: LLM forge with template fallback, moods, NPC memory ✓

## Gaps (the work)

1. **Trivia is 143 static Q&A pairs** (`easy.py:148`). No generation, no personalization, no anti-repeat across sessions. This is the #1 "feels static" offender — it's a flagship party game.

2. **No LLM question generation** for trivia/quiz anywhere. The case game has a template generator but trivia doesn't even have that.

3. **No cross-game personalization**: the interest profile from arena isn't used by game selection. "You play trivia most → suggest quiz duel" doesn't exist.

4. **No surprise mechanics**: GameMaster has moods but no "wild card" events, no dynamic twists mid-game.

5. **Hunt game**: daily hunt exists (from the screenshot: "daily hunt complete!") but need to check if quarry/targets are dynamic.

6. **Wordle/word games**: word lists — check size and whether they're static.

## Outside research (best-in-class patterns)
- **Jackbox**: prompts are the game — player-generated content > static banks. Lesson: let the LLM *and players* generate content.
- **AI Dungeon**: infinite LLM-driven narrative. Lesson: GameMaster narration should be able to *run* a game, not just narrate.
- **Geoguessr daily / Wordle daily**: one shared puzzle per day creates ritual. The daily hunt already does this — expand the pattern.
- **Hades "heat" system**: player-chosen difficulty modifiers. The `variants` dict is a start — expand to dynamic modifiers.
- **Anti-repeat done right**: the arena's persisted anti-repeat window is the pattern to copy to trivia/cases/words.

## Plan
1. `TriviaForge` — LLM-generated trivia (topic-personalized via interest profile) + seeded template fallback + persisted anti-repeat. Merge into `easy.py`'s TriviaRoyaleGame.
2. `QuizDuel` same treatment (shares the bank).
3. **Game recommender** — "because you play X" suggestions from play history.
4. **DM wild cards** — surprise twist events the GameMaster can inject (double-points round, traitor reveal, etc.).
5. **Word banks** — audit Wordle/Hangman/Anagram lists, expand + anti-repeat.
6. Tests for all. Commit section by section.
