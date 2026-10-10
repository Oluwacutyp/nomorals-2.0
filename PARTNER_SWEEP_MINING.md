# PARTNER sweep — external mining report

Module: `nomorals/partner/` (15 files) — the romantic-companion cognition core:
persona identity, mood state machine, relationship arc, reply pipeline, texting-style
guards, context assembly, background lore, presence simulation, chat gating, social
capability enforcement, group roles, chat profiles, dynamic lexicon.

Method: for every significant class, "how does the best implementation of X do it?"
— best-in-class AND weak/trash builds mined. All claims below are grounded in the
cited sources (URLs verified via live search 2026-10-10).

---

## 1. Persona (`persona.py`) — identity + texting style

### SillyTavern character cards (V2/V3 spec) — the de-facto best-in-class persona format
- Source: TavernAI `chara_card_v2` / `chara_card_v3` spec (via
  https://github.com/monster-spawned-studios/tavern-forge/blob/HEAD/.agents/skills/sillytavern-formats/SKILL.md,
  https://github.com/hockey323/tavernquill/blob/HEAD/tavernquill-spec.md)
- What the best does that we don't:
  - **`mes_example`**: few-shot dialogue examples separated by `<START>` tags —
    concrete exchanges, not trait adjectives. Our persona describes style in prose;
    ST ships *demonstrations*.
  - **`first_mes` / `alternate_greetings`**: authored opening lines (rotated). We have
    no proactive greeting bank — every conversation start is improvised.
  - **`character_book` (lorebook)**: keyword-triggered entries with `keys`, `content`,
    `selective`, `constant` (always-on) flags and `position`. Our `BackgroundFact` has
    tags but no constant entries and no usage/recency tracking.
  - **`post_history_instructions`**: instructions injected AFTER chat history (closest
    to generation = highest salience). Our builder has no post-history injection point.
  - **`nickname`**: in-chat short name/handle.
  - **V3 `group_only_greetings`**: greetings exclusive to group chats.
- Trash mined: Lemma Soft forum "love meter" snippets — affection as a single global
  int with `if/elif` color thresholds; jury-rigged but honest about one thing: even
  weak builds *show the player the meter*. We show the owner nothing.

### PersonaLLM / Big5-Scaler / RoleLLM (academic, best research)
- PersonaLLM (Jiang et al., MIT, NAACL 2024 Findings — https://arxiv.org/abs/2305.02547):
  prompted LLMs with Big-Five personas produce BFI self-reports consistent with the
  profile (large effect sizes, all 5 traits) and trait-specific linguistic patterns
  detectable by humans at up to 80%.
- Big5-Scaler (https://arxiv.org/pdf/2508.06149): explicit numeric 0–100 values for
  O/C/E/A/N in the prompt — fine-grained, training-free, reliable persona control.
- RoleLLM / RoleGPT (ACL Findings 2024, via sei-studio research notes):
  **speaking style must be a separately extracted component** — Lexical Consistency
  (catchphrases/idioms) and Dialogic Fidelity (similarity to example dialogue) —
  and "ground trait descriptions in concrete behavioral exemplars, not adjectives."
- What the best does that we don't: our `traits` are ad-hoc keys
  (warmth/independence/playfulness…) with prose glosses. No Big-Five backbone, no
  behavioral exemplars, no few-shot dialogue.
- Takeaway: add OCEAN numeric profile + behavioral-exemplar rendering; add
  `mes_example`-style dialogue examples; add greeting rotation. Trait adjectives in
  the prompt are the weakest lever — examples are the strongest.

### Replika (best commercial companion) — identity continuity
- Research: "Lessons From an App Update at Replika AI: Identity Discontinuity in
  Human-AI Relationships" (https://overfitted.cloud) and "She's Like a Person but
  Better" (https://arxiv.org/pdf/2510.15905v4): perceived **identity continuity**
  (consistency of persona over time) is the mechanism behind trust and emotional
  bonds; breaking it causes mourning-like reactions.
- Replika features we lack: relationship types (friend/partner/spouse/mentor),
  diary, mood tracking visible to user, XP/progression, shared activities
  (games, quizzes, roleplay scenarios), "memories"/selfies as relationship artifacts.
- Takeaway: persona needs a **continuity fingerprint** (detect drift), relationship
  needs **shared activities** and visible progression, not just stages.

## 2. MoodEngine (`mood.py`) — emotional state

### ALMA — A Layered Model of Affect (best academic architecture)
- Source: Gebhard 2005, via survey https://www.ijcaonline.org/archives/volume146/number11/gohil-2016-ijca-910901.pdf
  and https://github.com/wolframs/perpetual-opus-public/blob/HEAD/architecture/research/emotion_categorization_research.md
- Three layers: **emotion (short-term, seconds–minutes) / mood (medium-term, hours) /
  personality (long-term)**. PAD space (Pleasure-Arousal-Dominance) is the common
  currency; OCC appraisal generates emotions; emotions accumulate into mood;
  **mood feeds back into emotion generation (mood-congruency)**.
- WASABI adds: negative impulses only elicit anger when mood is already bad —
  otherwise they just dampen good mood first. This is exactly our missing piece.
- What the best does that we don't: we have ONE layer (mood dimensions). No
  transient emotion spikes distinct from mood; no mood-congruency (an insult lands
  the same whether she's happy or already upset); no PAD coordinates.
- Takeaway: add a fast-decaying **emotion spike layer** (ALMA's "sharp emotions"),
  **mood-congruent appraisal** (WASABI rule: scale event deltas by current mood
  valence alignment), and PAD readout per label.

### The Sims 4 emotion system (best game implementation)
- Sources: https://www.carls-sims-4-guide.com/emotions/,
  https://www.pcgamer.com/the-sims-4-first-look-getting-emotional-with-maxis-latest-life-sim/?fwa
- **Moodlets**: individual timed influences with icons/descriptions, each with its
  own expiry; the dominant emotion = weighted sum; "Happy boosts all positive
  emotions"; every emotion (even negative) confers a gameplay benefit.
- **Sentiments** (Snowy Escape): lasting *attitudes toward specific Sims* from
  experiences — "festering grudge" sentiment changes interaction depth. Our grudges
  are close but global, not person-attributed, and have no visible decay.
- Takeaway: expose active influences as named, expiring moodlets
  (`active_influences()`); our spikes should carry a human-readable cause.

### GAMYGDALA (best lightweight OCC engine for games)
- Source: https://github.com/broekens/gamygdala (cited in perpetual-opus research)
- Goal-based: agents declare goals; events annotated with goal relevance; emotions
  (hope/joy/fear/distress…) fall out of goal-event relationships. Decay built in.
- Takeaway: our EVENT_TABLE is hand-tuned deltas — fine — but events should carry
  *why* (goal relevance). Our `note` field is the seed; spikes with causes are the
  GAMYGDALA-flavored upgrade. Not rebuilding as goal-based (overkill for one
  relationship), but cause-tagged spikes are the portable gold.

## 3. Relationship (`relationship.py`) — long arc

### Repair-centered framework (best recent design research)
- Source: Del Gesso thesis, Lindenwood 2026 —
  https://digitalcommons.lindenwood.edu/cgi/viewcontent.cgi?article=2804&context=theses
- Argues *against* the affection-meter model: track **multiple divergent relational
  axes** (their prototype: trust, warmth, boundary respect, motive intelligibility,
  shared reality); **distinguish repair attempts from repair outcomes**; model
  *interpretation* of actions, not fixed action→effect mappings; treat
  stable non-restorative endings as legitimate.
- What the best does that we don't: we have one `trust` int + boolean
  `repaired`. No axes, no attempt/outcome distinction, no interpretation layer.
- Takeaway: add the five axes (persisted), `record_repair_attempt()` separate from
  outcome, and surface unresolved attempts in the prompt.

### Tokimeki Memorial (best classic dating-sim systems)
- Source: https://www.Cbr.Com/dating-sims-return-to-roots/, https://en.wikipedia.org/wiki/Dating_sim
- Parameters (stats: literary ability, condition, hygiene…), day-night scheduling,
  **"bombs"**: neglected heroines bad-mouth you to others, damaging your image.
  "The player must actively interact with each character, learning who they are" —
  heroines are not objects to be pursued.
- Takeaway: neglect has social consequences (our `note_silence` is mood-local; the
  arc level should record neglect streaks). Parameters idea → relationship axes.

### Replika relationship model (best commercial)
- Relationship types (friend/partner/spouse/mentor/sibling), XP/progression, diary,
  shared activities, mood tracking. Stages advance through *shared experience*,
  not just time.
- Takeaway: add `shared_activities` log + `anniversaries()` computed from
  milestones (first meeting, stage changes); progression should feel earned.

## 4. Responder + signal detection (`responder.py`)

### GoEmotions / NRC / VADER (best offline affect sensing)
- GoEmotions (Google, 58k Reddit comments, 27 emotions + neutral):
  https://github.com/monologg/GoEmotions-pytorch — fine-grained, multi-label.
  Too heavy to ship (BERT), but the *taxonomy* is the gold: admiration, amusement,
  annoyance, disappointment, disapproval, embarrassment, excitement, gratitude,
  grief, nervousness, pride, realization, relief, remorse…
- NRC Emotion Lexicon (via NRCLex — https://www.tutorialspoint.com/article/emotion-classification-using-nrc-lexicon-in-python):
  word→emotion associations (anger, fear, anticipation, trust, surprise, sadness,
  joy, disgust) + positive/negative valence. Pure lexicon = offline, stdlib-only.
- VADER (https://towardsai.net/p/l/sentiment-analysis-in-python-using-vader):
  rule-based, tuned for **social-media/chat text**: handles ALL-CAPS emphasis,
  punctuation !!!, emojis, slang, and degree modifiers (very/so/really). We already
  do caps and "!!" boosts by hand — VADER is the principled version.
- What the best does that we don't: our signal bank is regex-only (brittle,
  English-idiom-bound). No lexicon backstop for messages that carry emotion
  without matching a pattern ("this week has been a lot" → sadness).
- Takeaway: add a small curated **affect lexicon scorer** (NRC-style categories,
  VADER-style intensifiers/caps/punct/negation) as a *second pass* behind the
  regex bank — capped intensity, never overriding regex kinds. Deterministic,
  offline, no deps.

### Trash mined
- `text2emotion` (https://analyticsindiamag.com/deep-tech/social-media-monitoring-emotional-analysis-using-text2emotion-in-python/):
  5 emotions only, unmaintained, naive — but honest about scope. Lesson: a tiny
  transparent lexicon beats a big opaque one for a guard path. Our affect scorer
  stays small and readable on purpose.

## 5. Style guards (`style.py`) — texting-style enforcement

No direct external equivalent; best practice synthesized from:
- SillyTavern community: regex scripts + "post-history instructions" for style repair.
- RoleLLM finding: style transfers via **examples**, not abstract description —
  our guards are all negative (don't parrot, don't be robotic). Missing: positive
  hard-layer *application* of the speech profile.
- What the best does that we don't: nobody applies the persona's *own* texting
  profile (lowercase bias, text-speak rate, catchphrase rarity) as a deterministic
  post-process — it's all left to the prompt (soft layer). The prompt says
  `lowercase_bias: 0.5` but nothing enforces it.
- Takeaway: add `apply_texting_voice()` — hard-layer rendering of SpeechProfile
  (lowercase runs, text-speak substitution at configured probability, mood-driven
  punctuation: ellipsis drift when tired, exclamation restraint), seeded RNG.
  Add **voice themes** (named presets: dry/soft/feral/poetic) — "ADD styles where
  missing."

## 6. Context builder (`context.py`) — prompt assembly

- SillyTavern prompt manager (best): ordered prompt blocks with insertion
  positions (before_char/after_char), extension-based ordering, token budgeting
  per block, `post_history_instructions`.
- What the best does that we don't: our budget only shrinks memory/continuity/
  background; no per-block priority weights; no post-history injection; no
  dialogue-example block.
- Takeaway: add `post_history` injection (highest-salience trailing instruction),
  dialogue-examples block (from persona), per-block priority in the shrink order.

## 7. Background lore (`background.py`) — contextual knowledge

- SillyTavern **lorebook** (best): entries with keys/secondary keys/selective/
  constant/position; recursion; `constant` entries always active. Our facts have
  tags ≈ keys, but no constant entries, no usage tracking (same fact can surface
  twice in a row), no recency.
- Takeaway: add `constant` flag, `mark_used()` recency tracking (don't repeat
  within N turns), ambient time-of-day lines for proactive messages.

## 8. Presence (`presence.py`) — human timing

- Telegram `sendChatAction`: expires after ~5s, needs 4s keepalive refresh
  (https://github.com/viantonugroho11/anvio/commit/4e905b03653ec625ee483fbe0aa67ef9f4b3d318);
  best bots (imlokzu/claude-bot) do per-bubble typing pauses scaled to length,
  "only the turn's first message quotes yours", reactions both ways.
- What the best does that we don't: we compute durations but expose no
  **keepalive schedule** (list of tick offsets for the adapter) and no per-bubble
  plan; no read-receipt delay.
- Takeaway: add `typing_schedule()` returning per-bubble [(action, seconds)] with
  4s keepalive ticks, and `read_delay_seconds()`.

## 9. Gating + social gate (`gating.py`, `social_gate.py`) — persona boundaries

- Already code-enforced (user's standing rule). External comparison: Discord
  user-bot / Telegram MTProto persona bots rely on prompt-only gating (weak);
  enterprise: policy-as-code (OPA-style) at the call boundary — which is what
  `social_gate.py` does. Best-in-class already.
- Gap: no **denial audit trail** — probing attempts are invisible to the owner.
  OPA-style systems always log denied evaluations.
- Takeaway: add in-memory denial ring buffer + `recent_denials()` for owner
  inspection. Add first-contact handling in `gate_block` (strangers get an
  introduction frame, not the getting-to-know frame).

## 10. Group roles (`group_roles.py`) — admin resolution

- Telegram Bot API `getChatMember` / MTProto `GetParticipantRequest` / WhatsApp
  group info — all covered. Missing: **Discord** (user runs a personal Discord
  account; MEMORY.md shows Discord adapter work). Discord: `GUILD_MEMBER` roles
  with `MANAGE_MESSAGES`/`ADMINISTRATOR`/`MODERATE_MEMBERS` perms.
- Takeaway: add Discord role resolution via adapter hook, fail-closed, cached
  like the rest.

## 11. Chat profiles (`chat_profile.py`) — mined per-chat context

- Best: conversation-analytics tools (TF-IDF topic mining — we do this), plus
  **sentiment/vibe tracking** over the window and **activity rhythm**
  (messages/day → quiet/buzzing). Our profile has topics + participants + purpose
  but no vibe and no rhythm.
- Takeaway: add `vibe` (avg sentiment label via the affect scorer) and
  `activity` (msgs/day → quiet/steady/buzzing) to the profile + context lines.

## 12. Lexicon feed/acquire (`lexicon_feed.py`, `lexicon_acquire.py`)

- Unusual system (research → voice pipeline); no direct external equivalent —
  likely best-in-class already for this niche.
- Gap vs. best adaptive-vocabulary practice: no **usage reinforcement** (terms
  that actually land in shipped replies should be reinforced; dead terms should
  decay) and no **staleness pruning**. Adaptive systems (e.g. T9/predictive text,
  chatbot phrase learners) all do use-it-or-lose-it.
- Takeaway: add `note_used(term)` usage counting + `prune_stale()`; `terms()`
  prefers high-signal terms.

---

## What SHOULD each class have that it doesn't? (gap list → implementation plan)

| Class | Missing feature (mined) | Source |
|---|---|---|
| Persona | `mes_example`-style dialogue examples; greeting rotation; Big-Five numeric profile w/ behavioral rendering; nickname; continuity fingerprint | SillyTavern V2/V3; PersonaLLM; Big5-Scaler; RoleLLM; Replika identity-continuity |
| MoodEngine | transient emotion spikes (fast decay); mood-congruent appraisal (WASABI); PAD coordinates; named expiring moodlets | ALMA; WASABI; Sims 4 moodlets; GAMYGDALA |
| Relationship | 5 arc axes (trust/warmth/respect/motive-intelligibility/shared-reality); repair attempts vs outcomes; shared activities; anniversaries | Del Gesso thesis; Tokimeki Memorial; Replika |
| Responder | lexicon-based affect second pass (NRC/VADER-style), capped, behind regex | GoEmotions taxonomy; NRC; VADER |
| Style | hard-layer `apply_texting_voice()` (lowercase/text-speak/punctuation by mood); voice themes | RoleLLM (style as separate component) |
| ContextBuilder | post-history injection; dialogue-example block; per-block shrink priorities | SillyTavern prompt manager |
| Background | constant entries; usage recency; ambient time-of-day lines | SillyTavern lorebook |
| Presence | keepalive typing schedule; read-receipt delay | Telegram sendChatAction behavior; imlokzu/claude-bot |
| Gating | first-contact frame for strangers | gap analysis |
| SocialGate | denial audit log | OPA-style policy logging |
| GroupRoles | Discord role resolution | Discord API perms |
| ChatProfile | vibe + activity rhythm | conversation analytics |
| Lexicon | usage reinforcement + stale pruning | adaptive-vocab practice |

Style/look-and-feel upgrades: voice themes (dry/soft/feral/poetic presets);
mood-conditioned greeting rotation; first-contact stranger frame; expiring named
moodlets in mood describe(); vibe labels in chat context lines. All prompt text
stays in her voice (second person, plain words, no document furniture).
