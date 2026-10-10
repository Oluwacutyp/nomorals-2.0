# BOOKS Sweep — External Mining Report

Mined 2026-10-10. Every significant class in `nomorals/books/` researched
against the best implementations OUTSIDE the repo (best-in-class tools,
open-source projects, documented craft systems). Gold is merged, never
copied; deltas listed per class below.

## 1. bible.py — StoryBible / BibleBuilder

**Gold: NovelAI Lorebook.** Entries = (Entry Text + Activation Keys). Keys are
case-insensitive by default, regex when wrapped in `/.../ `, `&` = AND-logic
(multi-key activation), Always-On entries injected every prompt, Enabled toggle,
Hidden entries, Lore Generator, import/export (even embedded in PNG), Placement
tab, Phrase Bias. Keyed injection solves long-story amnesia: facts surface only
when mentioned.

**Gold: Novelcrafter Codex.** Auto-mention indexing (names/aliases linked as you
type), Global Mapping (every appearance across the manuscript), Progressions
(timeline-tracking: character/world details change over the story; outdated lore
overwritten), Smart Highlighting (overused words, crutch phrases, dialogue-tag
control, AI-pattern detection), codex feeds every AI operation as context.

**Gold: AI Dungeon context stack.** Canonical assembly order: AI Instructions →
Plot Essentials (always-on facts) → Story Cards (triggered) → Story Summary →
Memory Bank (embedding retrieval) → History → Author's Note → Last Action →
buffer tokens. Lessons: always-on must be TRIMMED (stale facts actively steer
wrong); Author's Note is short and by-position; Story Summary is auto (manual
edits cause drift); cards must not restate plot essentials.

**Trash mined:** flat JSON "character lists" with no keying, and prompt-bombs
that dump the whole bible every chapter (context bloat, no retrieval).

**Deltas for bible.py:**
- New `LorebookEntry` dataclass: name, type (character/location/item/concept/
  rule), text, keys (with `/regex/` + `&`-AND support), always_on, enabled,
  hidden, order, aliases.
- `StoryBible.inject(recent_text, max_chars)` → AI-Dungeon-order assembly:
  always-on facts + key-triggered entries (scan recent text), with chain
  activation and per-entry char budget.
- `BibleBuilder.build` auto-derives activation keys + aliases per character.
- `mention_index(chapters)` — Novelcrafter-style: every character/place mention
  mapped to chapter numbers (global mapping).
- `Progression` entries: timeline-tracked detail changes
  ("Aiko: scarred ← ch.12").
- `export()`/`import_()` JSON round-trip; `consistency_check()` flags stale
  entries (always-on bloat guard).
- `StoryBible.brief()` gains a compact injectable form.

## 2. fiction.py — FictionWriter / GenreEngine (+6 genre engines)

**Gold: Sudowrite Story Engine.** Pipeline: brain dump → synopsis → characters
(mentors/rivals/friends generated from synopsis) → act/chapter outline →
per-chapter BEATS (step-by-step instructions guiding the AI to write the
chapter) → prose generation → human edit. Beats are the atomic unit; the model
writes from beats, not vibes.

**Gold: Save the Cat (Blake Snyder; novel adaptation by Jessica Brody).**
15 beats mapped to % of book: Opening Image (0-1%), Theme Stated (~5%),
Setup (1-10%), Catalyst (~10%), Debate (10-20%), Break into Two (~20%),
B Story (~22%), Fun and Games (20-50%), Midpoint (~50%), Bad Guys Close In
(50-75%), All Is Lost (~75%), Dark Night of the Soul (75-80%), Break into
Three (~80%), Finale (80-99%), Final Image (99-100%). Single-scene vs
multi-scene beats. Hero's Want/Flaw/Need drive which events happen.

**Gold: Character psychology (Truby/Weiland).** Five-point core: Want (concrete
external goal) vs Need (unconscious requirement, in tension with Want) → arc;
Wound (specific, dateable event); Lie (false premise drawn from the wound);
Ghost (sensory shorthand of the wound that resurfaces under pressure). No arc
without Want/Need tension.

**Trash mined:** keyword-swapping "genre engines" that only change adjective
pools; outlines that hand the model a chapter number and pray.

**Deltas for fiction.py:**
- New `BeatSheet`: STC 15 beats → chapter positions for any chapter count;
  `beat_for(chapter_no)` returns the active beat + genre-colored instruction.
- `GenreEngine.chapter_brief` upgraded to emit Sudowrite-style scene beats
  (3-6 step instructions), style/POV directive, and want/need tension cue.
- `GenreEngine.scene_beats(brief, n)` splits a chapter brief into scene-level
  beats.
- `GenreEngine.voice()` hook: POV + tense + tone directive per genre.
- `_cast()` enriched: want/need/wound/lie/ghost/archetype per character
  (heuristic template + model-fill path).
- New `XianxiaEngine` (cultivation realms, sects, face, breakthroughs,
  young-master/rival/elder cast) — the webnovel genre our engines lacked.
- `FictionWriter.write_chapter` accepts `beats=` override and injects
  bible lorebook text into the model prompt (NovelAI-style keyed context).

## 3. continuation.py — StoryContinuer

**Gold: AI Dungeon context assembly order** (see §1) + **takes/retries**:
every story part can hold multiple alternative takes; the user picks the live
one; writing below a non-live take starts a cheap branch. Also **Memory Bank**:
auto-embedded memories retrieved by similarity; **Auto-Summary**: rolling
summary the model maintains.

**Gold: 1667 (NovelAI-successor) Facts.** `always` Facts (in every request) vs
`keyed` Facts (enter on key match), secondary AND/NOT keys, scan depth, chain
activation — the distilled lorebook idea.

**Trash mined:** continuations that feed "last 3500 chars + write more" —
no memory, no style lock, no alternates; quality collapses by chapter 5.

**Deltas for continuation.py:**
- `_assemble_prompt(bible, chapter_no, style, recent, summary)` in canonical
  AI-Dungeon order: instructions → always-on bible facts → keyed lorebook hits
  → rolling summary → style profile → recent tail → author's-note-style voice
  lock → buffer note.
- `takes` parameter: generate N alternative continuations, score them
  (voice-match + hook strength + no-repeat), keep all takes on disk, select
  best as live.
- Rolling summary: after each chapter, append a compressed recap to the
  bible (auto, with manual-edit drift warning in docstring).
- Style lock: bible style profile carried as an explicit "Author's Note"
  block near the prompt tail.

## 4. sources.py — SourceAdapter + Fetcher

**Gold: FanFicFare (~110 adapters).** Central pattern: `BaseSiteAdapter` with
`getSiteDomain/getSiteURLPattern`, `extractChapterUrlsAndMetadata()`,
`getChapterText(url)`; automatic adapter registration via `adapters/__init__`
(domain → class map). Fetchers: requests, cloudscraper (CloudFlare bypass),
proxy fetchers; page caching; `SleepDecorator` rate limiting;
`ProgressBarDecorator`; chapter-title patterns (`${number}. ${title}`);
`mark_new_chapters`; metadata-first design (status, genres, tags, dates).

**Trash mined:** per-site scrapers with no shared base, no retry, no rate
limit, hardcoded single CSS selector per site (breaks on every redesign),
"generic" fallbacks that grab nav/footer as chapter text.

**Deltas for sources.py:**
- `ADAPTERS` auto-registry: `register_adapter`, `adapter_for_url(url)`
  (domain-pattern matching), `list_adapters()`.
- `Fetcher`: exponential-backoff retry, configurable `sleep_between`
  rate limit, rotating User-Agents, per-host cookie jars (exists) + request
  logging for debugging.
- `SourceAdapter.download_story(url, on_chapter=None)` — metadata +
  chapter list + full chapter fetch with progress callback and resume
  (skip already-cached chapter URLs).
- Richer `StoryMeta`: status (ongoing/completed/hiatus), genres/tags,
  rating, cover_url, last_updated, word_count estimate; `fetch_cover()`.
- URL normalization (strip tracking params, canonical chapter URLs).

## 5. library.py — Library

**Gold: Calibre.** `metadata.db` (SQLite) central catalog; tags, series
(series + series_index), ratings, custom columns, identifiers (ISBN etc.),
full-text search plugin ecosystem, virtual libraries/shelves.

**Gold: KOReader.** Per-page-turn statistics (duration rows → reading sessions,
calendar views, streaks), bookmarks + highlights with styles + notes, export
to text/markdown/HTML/JSON, sync to Readwise/Joplin, book map, vocabulary
builder, reading progress sync across devices.

**Trash mined:** libraries that are just a folder listing; "progress" = one
float with no history.

**Deltas for library.py:**
- Reading sessions: `start_session`/`end_session` → per-book rows
  (started, seconds, chapters, words) → WPM, total time, calendar.
- Reading streaks: consecutive-day reading from sessions.
- `export_annotations(slug, format="markdown")` — KOReader-style export of
  bookmarks + notes with chapter refs and quotes.
- `series` support: `set_series(slug, name, index)`, `list_series()`.
- `stats(slug)` → sessions, total minutes, WPM, streak, % complete.
- `currently_reading` smart shelf (any book with a session in last 7d).

## 6. reader.py — StoryReader

Same KOReader gold + web-serial update patterns.

**Deltas:**
- `check_updates(slug="")` → compare cached chapter count vs live
  `chapter_list` → new-chapter report (FanFicFare `mark_new_chapters` idea).
- Chapter annotations: highlights + notes per followed chapter (KOReader
  style), `export_annotations(slug)`.
- `reading_stats()` — sessions/streak lite for followed stories.
- `catch_up(slug)` — read all unread chapters in order, advancing progress.

## 7. branches.py — StoryBranch

**Gold: inkle Ink.** Weaves (branching choice structures), gather points
(collapse branches back), knots/stitches (named sections), threads (parallel
strands), tunnels (subroutine calls), conditional text, variables, tags,
shuffle/cycle/sequence alternates. Markup-first: text is primary, logic
embedded.

**Gold: ai-dnd story tree.** Story is a TREE: any turn holds multiple takes;
forks are cheap (~100 bytes, borrow ancestors); branch panel with rename/
delete; visual tree ("see the tree"); switching restores state.

**Trash mined:** "branches" that are just copied folders with no relationship
metadata; merges that are copy-paste.

**Deltas:**
- Choice nodes: `add_choice(after_chapter, prompt, options)` — interactive
  decision points with reader picks recorded.
- `tree()` — ASCII/structured branch graph (fork points, choice nodes,
  chapter counts, statuses).
- `diff(branch)` — chapter-by-chapter divergence summary vs canon
  (word counts, first-divergence chapter).
- `rename()`, richer merge strategies (`append` | `replace-from` |
  `interleave`), merge writes a merge record (what came from where).

## 8. publish.py — SerialPublication

**Gold: Royal Road author practice.** Consistency beats volume ("new chapter
Tuesday" > "four a week"); write AHEAD — the backlog is the real schedule,
calendar is what readers see; Royal Road's own scheduler posts drafts at a
fixed minute; 2-4k word chapters; launch with ~10 chapters day one; author
notes for schedule/stub alerts; shoutout swaps; track followers (views are
noisy); never release on the hour (front-page competition).

**Trash mined:** drip-schedulers with no backlog concept (miss one day, dead
streak) and no reader analytics.

**Deltas:**
- Cadences: `daily`, `weekly`, plus `weekly:mon,wed,fri` style weekday
  schedules ("new chapter Tuesday").
- Backlog buffer: `buffer_status()` — chapters written ahead vs released;
  warn when buffer < 2.
- Author notes attached to releases (`set_author_note`).
- `launch_plan()` — day-1 bulk release helper (first N chapters at once).
- `retention_curve()` — per-chapter feedback/views → drop-off analysis
  (which chapter lost readers).
- `next_releases(n)` preview of upcoming release datetimes.

## 9. collab.py — CriticAgent / CollaborativeSession

**Gold: Sudowrite feedback plugins** (Simulated New Yorker Review, Brutal
Honesty Bot, Encouraging Writing Buddy) — persona-driven critique with
distinct voices and standards.

**Gold: Novelcrafter Smart Highlighting.** Overused words, crutch phrases,
repetitive metaphors, dialogue-tag audit (distinct character voices),
AI-pattern flagging.

**Gold: character psychology (§2).** Character sheets with want/need/wound/
lie/ghost + voice profile (speech tics, vocabulary band, rhythm).

**Trash mined:** single-paragraph "looks good!" reviewers; critique that
ignores the story bible.

**Deltas:**
- New `StyleAnalyzer`: filter-word scan (saw/felt/heard/knew/realized —
  show-vs-tell markers), crutch-phrase frequency, sentence-length variance,
  dialogue-tag audit, repeated-word heat, paragraph rhythm. Pure-Python,
  no model needed.
- New `CharacterSheet`: want/need/wound/lie/ghost/archetype/voice +
  `voice_check(dialogue)` scoring whether a line sounds like the character.
- `CriticAgent` personas: `editor` (structure), `line` (prose),
  `beta` (reader feel), `brutal` (no mercy). `review()` accepts persona;
  style analysis always runs as a first pass; bible continuity check second.
- `CollaborativeSession.cast_character` stores full sheets; `write_scene`
  injects the character's voice profile + lorebook hits.

## 10. write.py — chapter composers + Expand/Describe/Rewrite

**Gold: Sudowrite primitives.** Expand (grow a beat into a scene), Describe
(sensory description from a noun phrase), Rewrite (rephrase with direction),
Canvas (section-level drafting).

**Deltas:**
- `expand(text, context, *, target_words)` — beat → scene (model when
  available, structured heuristic when not).
- `describe(subject, context)` — sensory description generator.
- `rewrite(text, instruction, context)` — directed rewrite.
- `suggest_hooks(prev_tail, context)` — 3 candidate closing hooks.
- All accept `suggest=` callable (model) with real heuristic fallbacks —
  never stubs.

## 11. forge.py — BookForge build

**Gold: EbookLib.** EpubBook + metadata (identifier/title/language/authors) +
EpubHtml chapters + EpubNav/EpubNcx + CSS + spine; `write_epub`.

**Gold: Deckle.** Markdown → finished EPUB/PDF/Word on the author's machine.

**Key insight:** EPUB is a ZIP with a fixed skeleton (mimetype, container.xml,
content.opf, toc.ncx/nav.xhtml, chapters as XHTML). Zero-dependency EPUB3
writer in stdlib `zipfile` is fully viable — matches the standing
best-free-deps rule better than requiring ebooklib.

**Deltas:**
- `build_epub(slug, cover=None)` — hand-rolled valid EPUB3 (stdlib only):
  title page, cover, TOC nav + NCX, chapter XHTML, Dublin Core metadata
  (title/creator/language/identifier/date), CSS stylesheet. Optional
  ebooklib path if installed (richer).
- `build_html(slug)` — single-file styled HTML book (great for phone
  reading/sharing).
- `build()` gains `formats=("pdf",)` → `("pdf","epub","html")`.
- Book model gains `author`, `language`, `series`, `cover` metadata
  (backward-compatible `from_dict`).

## 12. outline.py

**Gold: Save the Cat beat sheet (§2).**

**Deltas:**
- `beat_sheet_outline(book, n_chapters)` — maps the 15 STC beats to chapter
  numbers by % position; returns chapters with beat annotations.
- `three_act_map(n_chapters)` — act boundaries at 25%/50%/75% (STC-aligned).
- `make_outline` accepts `structure="beats"` to use it.

## 13. tools.py + styles.py (NEW)

**Gold:** Royal Road/Tapas chapter presentation; KOReader progress display.

**Deltas:**
- New `styles.py`: output themes (`rich` emoji/typographic, `plain`
  ASCII-safe, `minimal`); `render_book_card`, `render_chapter_card`,
  `render_progress_bar`, `render_stats_table`, `render_beat_sheet`,
  `render_tree` (ASCII branch graph).
- tools.py: new registrations — `book_build_epub`, `book_build_html`,
  `story_lorebook` (inject preview), `fiction_beats` (beat sheet for a
  story), `branch_tree`, `branch_choice`, `publish_launch_plan`,
  `publish_retention`, `library_stats`, `library_export`, `reader_updates`,
  `collab_stylecheck`, `character_sheet`, `write_expand`, `write_describe`,
  `write_rewrite`, plus `display` (pretty card) fields on status outputs.

## 14. model.py

- `Book`: add `author`, `language`, `series`, `series_index`, `cover_image`
  (all optional, `from_dict` defaults); `progress_pct()`;
  `Chapter`: `status` field (draft/revised/final), `takes` (alternate
  versions), `word_target`.
