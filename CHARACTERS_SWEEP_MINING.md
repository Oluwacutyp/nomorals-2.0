# CHARACTERS Module — External Mining Report

**Module:** `nomorals/characters/` (14 files, ~2,360 lines)
**Date:** 2026-10-10
**Method:** mine best AND trash outside the repo, then merge gold into the real classes.

---

## Sources mined

| # | Source | What it is | Gold taken |
|---|--------|-----------|------------|
| 1 | Park et al., **Generative Agents: Interactive Simulacra of Human Behavior** (Stanford/Google, UIST 2023, arXiv:2304.03442) + open-source code | The canonical character-agent architecture | Memory stream, **reflection** (periodic LLM synthesis of observations into higher-level insights — ablation showed it's *critical* for believable behavior), three-factor retrieval (recency × importance × relevance), recursive planning |
| 2 | **Inworld AI Character Engine** (Inworld/NVIDIA docs) | Best-in-class commercial NPC platform | Three layers (Character Brain / Contextual Mesh / Real-Time AI); character fields: motivations, **insecurities**, stage of life, history, emotional tendencies, interests; "Goals and Actions" triggers; **Relationships** feature — allies turn enemies based on treatment |
| 3 | **Façade** (Mateas & Stern, 2005) — drama management research | Foundational interactive drama | **Drama manager** + story beats: an invisible agent that proactively adds/retracts behaviors and discourse contexts; beats with preconditions/effects; affinity games; canonical conversational sequences reshaped by metabehaviors |
| 4 | **Mem0 / Zep/Graphiti / Letta / MIRIX** | Modern agent memory systems | Episodic/semantic/procedural tiers (MIRIX: 6 components), hybrid retrieval (vector+graph+BM25), supersedence pointers with provenance, background "sleep-time" consolidation |
| 5 | **Big Five / OCEAN in dialogue agents** (Big5-Scaler paper, PersonageNLG, MDPI Electronics 2026 study) | Personality science in character AI | Explicit **numeric OCEAN values (0–100) in prompts** reliably elicit personality-consistent behavior; personality shows at lexical/syntactic/semantic levels; personality-consistency can be *measured* (11-point evaluator scale) |
| 6 | **Stylometry / authorship attribution** (survey literature: Stamatatos, Grieve, Burrows' Delta) | Voice-fingerprint science | Best discriminators: **function-word rates** (unconscious, hard to fake), vocabulary richness (type-token ratio), word-length distribution, sentence-length patterns; content words are topic-contaminated — *style must exclude topic* |

---

## Class-by-class: how the best do it vs. what we have

### `Character` (character.py) — the agent entity
- **Stanford:** agents have *reflection* — without it, characters stay flat (ablation-proven). We have zero reflection. → ADD `reflect()`.
- **Inworld:** characters carry insecurities, stage of life, emotional tendencies, interests. We have persona traits only. → ADD fields.
- **Inworld:** "Goals and Actions" — goals trigger behaviors. Our goals are static strings with no lifecycle. → ADD goal states (pursuing/achieved/paused).
- **Stanford retrieval:** recency × importance × relevance. Our `recall` is word-overlap only. → UPGRADE to three-factor scoring.
- **Big5-Scaler:** numeric OCEAN conditioning in prompts beats free-form adjectives. Our persona is free-form only. → ADD `ocean` mapping + prompt encoding.
- **Secrets:** we say "may slip under pressure" but there is no pressure mechanism. → ADD slip thresholds.
- **Style gap:** `_fallback_line` returns `[Zara — witty] context...` — a bracketed stub, not a voice. → UPGRADE to persona-flavored fallback lines.

### `CharacterContextBuilder` (context.py) — prompt assembly
- Authority ordering (identity → state → memory → contract) matches best practice; budget handling is good.
- **Gaps:** no goals-as-agenda block, no OCEAN encoding, no insecurities, no relationship-with-the-*other*-character block for ensembles, no voice-drift note. → ADD all.

### `CharacterMemory` (memory.py) — memory view
- Flat fallback mirrors `Character.recall`; good degraded path.
- **Gaps:** no importance-weighted three-factor scoring in fallback, no dedupe/consolidation, no reflection writes. → ADD consolidation + align scoring with Stanford.

### `Dialogue` / `converse` / `character_initiate` (dialogue.py)
- Solid turn loop; initiation exists but is called manually.
- **Inworld/Stanford gold:** characters act *proactively* — reach out driven by motives/mood/inactivity, not just when invoked. → ADD `proactive_pulse()` (decides *whether* to reach out; returns None most of the time — restraint is the feature).

### `Scene` / `run_scene` / `podcast_episode` (ensemble.py)
- **BIGGEST GAP — Façade:** we have scenes with NO drama management. Façade's whole contribution is the drama manager that watches energy, steers beats, reshapes discourse. Our scenes are round-robin; energy can flatline, nobody steers. → ADD `DramaDirector` with beats (precondition/effect), energy tracking, steering interventions.
- **Style gap:** one scene format. Façade uses canonical conversational sequences; podcasts/interviews/debates/roasts have different shapes. → ADD scene format presets.
- Podcast host-steers every 3rd turn — good; generalize via director.

### `run_agent_match` (match.py) — game seats
- Real engine path — good. **Style gap:** matches start cold. Game-night culture = banter, rivalries, callouts. Relationship graph has rivalry data we never use at the table. → ADD pre-game banter intro (persona + edge-aware), rivalry callouts.

### `RelationshipGraph` / `Edge` (relationships.py) — directional 5-dim edges
- Strong vs. the field: most game systems use a single affinity number; five slow-moving directional dims beats it.
- **Inworld gold:** ally↔enemy *transitions* are announced events. Our `_recompute_kind` flips silently. → ADD transition detection + narrative line ("Zara and Kilo are rivals now").
- **Façade gold:** story values / social games. → ADD `story_of(a, b)` — the relationship as a short narrative, plus triadic helper (friend-of-friend tension).
- EVENT_DELTAS lacks: secret sharing, defending someone, abandonment, celebrating, forgiveness. → ADD.

### `arcs.py` — beliefs/milestones
- Belief confidence+revisions is good (matches "scar tissue" framing).
- **Stanford gold:** reflection trees — reflections that feed later reflections. → ADD reflection records that recall surfaces.
- **Style gap:** `arc_summary` is bullet text. → ADD narrative rendering ("the story of Zara so far").

### `cast_for` / `chemistry` (casting.py)
- Good fit + chemistry greedy ensemble. **Gaps:** no negative casting ("who must NOT be in this room"), no rotation/anti-repeat for recurring shows, no audience-aware casting. → ADD.

### `CharacterStore` (store.py) — JSON persistence
- Solid, atomic writes. **Gaps:** no search by trait/skill/role (casting always loads everything then filters — fine at 6 chars, not at 60). → ADD `find()` query + `export_all()`.

### `voice.py` — voice fingerprint
- Shallow lexical+structural — honest scope. **Stylometry gold:** content-word top-lists are topic-contaminated; **function-word rates** are the robust discriminator. We have zero function-word features, no vocabulary richness (TTR), no word-length distribution. → ADD function-word profile, TTR, word-length stats; `style_report()` god-tier rendering.

### `process_session` (processing.py) — background pass
- Musubi-style bookkeeping — good. **Gaps:** no reflection generation (Stanford's critical piece) after salient sessions; no voice-drift flagging (we compute fingerprints but never *use* consistency). → ADD reflection hook + drift report in return dict.

### `seeds.py` — six starter characters
- Rich backstories, secrets, catchphrases — strong. **Inworld gold missing:** insecurities, stage of life, OCEAN profiles, interests. → ENRICH all six. (No new seeds — the six-way relationship mesh is carefully balanced.)

### Presentation (NEW)
- **No presentation layer exists.** Everything renders as plain strings. Task demands god-tier feel: character cards, cast announcements, styled transcripts, relationship web diagrams, arc stories, voice style reports. → ADD `render.py` with themes (rich/minimal).

---

## What "trash" taught us
- Flat character cards (static prompts) → sycophancy under pressure (MnemoLink comparison doc). Our `spine` + anti-sycophancy instruction is the right defense; OCEAN conditioning strengthens it.
- Overwriting memory on contradiction loses history; additive wins (Zep supersedence pointers). Our additive beliefs/memory already do this — keep.
- Content-word voice matching = topic matching, not voice matching (stylometry). Our `top_words` risks this; function-word profile fixes it.
