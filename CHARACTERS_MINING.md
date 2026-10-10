# Phase 4: Characters — Mining Report

## What's already strong (don't rebuild)

### Character system (`nomorals/characters/`, ~1,800 lines)
- `Character` dataclass: persona traits (0–1), mood (valence/arousal/trust), beliefs with confidence + revision count + history, secrets (never volunteered, may slip), core_motive, expression (speech patterns, catchphrases, emoji habits), spine (anti-sycophancy 0–1), skills, roles
- `arcs.py`: belief formation/challenge/drop with scar-tissue semantics; `grow_from_interaction`
- `dialogue.py`: brain↔character conversation, character-initiated
- `ensemble.py`: multi-character scenes, relationship-aware speaking
- `casting.py`: situation-based casting with chemistry scoring (complementary energy, shared humor)
- `relationships.py`: RelationshipGraph with warmth/respect/familiarity/friction dimensions
- `processing.py`: post-session background pass — updates mood, relationships, beliefs, salience
- `seeds.py`: 6 deep starter characters (Zara, Elder, etc.) with inter-relationships
- `store.py`: JSON persistence
- Spine-wired via `nomorals/tools/characters.py`

### Partner system (`nomorals/partner/`, ~3,700 lines)
- 10-dimension mood engine with hysteresis labeling, circadian decay, fights/grudges
- Relationship stages (acquaintance→established), milestones, fights, user profile
- Context builder: strict authority-ordered prompt assembly, token-budgeted
- Speech profile with emoji rate, catchphrases, text-speak
- Style guard, gating, group roles (just built)

### Memory system (`nomorals/memory/`, ~6,000 lines)
- Unified MemoryManager: episodic/semantic/working tiers, FTS + vector hybrid recall
- Contradiction strategy chains (negation/preference/value/LLM adjudication), additive resolution
- Scheduled additive consolidation, proactive recall, distillation
- Delivery scoring (when to surface, not just what), knowledge-state novelty
- User model (person-shaped materialized view), people graph, routines

## The gaps (what this phase builds)

### 1. Character memory is flat — the big one
`Character.memory` is a plain list of `{ts, text, salience}` dicts with word-overlap recall. It's completely disconnected from:
- MemoryManager (no episodic/semantic tiers, no FTS, no vectors)
- Contradictions (characters never notice when they contradict themselves)
- Proactive recall (characters never surface memories unprompted)
- Delivery scoring (characters mention things at the wrong time)
- Embeddings (no semantic recall)

**Build:** give each Character an optional full memory backend — MemoryManager scoped to the character — with the flat list as fallback. Characters get contradictions, proactive recall, and delivery scoring for free.

### 2. No voice distinctness enforcement
Characters have `expression` (catchphrases, emoji habits) but nothing verifies two characters actually sound different, or that a character stays in voice across sessions. The partner has a style guard; characters don't.

**Build:** voice fingerprint per character (lexical markers, sentence-length distribution, emoji rate, question rate) computed from their dialogue history; a consistency scorer that flags drift; used in the post-session processing pass.

### 3. Character ↔ partner bridge is one-way
The partner system (mood, relationship, context) is built for ONE relationship (Devon↔owner). Characters have their own mood/relationship systems that don't feed into the partner pipeline when a character talks to the owner.

**Build:** when a character speaks to the owner in chat, route through a character-aware context builder — character persona + mood + relationship-with-owner, rendered with the same authority ordering as the partner builder.

## Non-goals
- Rebuilding the partner mood/relationship engine (it's good)
- Rebuilding MemoryManager (it's good)
- New character seeds (6 is enough; owner can add)

## Build order
1. Character memory backend (MemoryManager per character)
2. Voice fingerprint + consistency scoring
3. Character-aware context builder for owner chat
4. Tests, commit, push
