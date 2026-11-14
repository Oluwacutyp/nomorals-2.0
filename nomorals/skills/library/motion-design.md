---
id: motion-design
title: Motion design — Disney principles, timing and easing, choreography
summary: How motion is designed in Devon: the Disney 12 principles adapted for UI, duration and easing tables, motion personalities, choreography and stagger rules, and the quality bar every animation must clear. Ported from the LottieFiles motion-design skill (MIT).
topics: motion, animation, easing, timing, choreography, transitions, micro-interactions
triggers: animate, animation, motion design, easing, transition, micro-interaction, loading state, page transition, scroll animation, choreography
layer: media
version: 1
---

# Motion design

This skill is Devon's motion director. It applies whenever she makes
something move: a UI animation on one of her web surfaces, a
micro-interaction, a loading or success state, a page transition, a
scroll-triggered effect, a Lottie composition, or a scene in her
motion engine. Motion is never decoration here — it is feedback,
guidance, and brand character, and it is judged by the same bar
every time.

## Where it lands in Devon

- **Web surfaces** (the handoff link page, the pairing page, the
  computer organ): CSS animations and transitions. This is where
  the easing tables and spring values below are spent directly.
- **Video** (`nomorals/media/edit_engine/keyframes.py`,
  `nomorals/media/edit_engine/transitions.py`,
  `nomorals/media/motion_studio/`): her keyframes carry only three
  easings today (`_ease_in`, `_ease_out`, `_ease_in_out`, plus
  "smooth"), and her video transitions are cut/fade/flash/dip/
  whip_pan/glitch_cut. The tables below are the palette those
  easings are chosen from — when a scene needs a curve that is not
  one of the three, that is a gap to close in the engine, not a
  reason to ship a linear move.
- **Lottie** (`nomorals/social/chat/emoji_anim.py`): she builds
  Lottie compositions natively. The principles below govern their
  timing the same way they govern CSS.
- **Springs**: for sampled CSS springs, see the `motion-web` skill
  (Motion ai-kit). For real spring physics in code, the spring
  parameter table below is the starting point.

## When to apply

Use this skill when creating UI animations (buttons, cards, modals,
page transitions), micro-interactions and feedback animations,
loading, success or error states, animated illustrations, or
scroll-triggered and progress-based animations; when establishing a
brand motion identity; and when choreographing multi-element
sequences.

**Decision tree:**
1. Does it serve a functional purpose (feedback, guidance)? →
   Timing rules for responsiveness.
2. Does it express brand personality? → Motion personality
   archetypes.
3. Does it tell a story or guide attention? → Disney principles +
   choreography.
4. Is this a complex multi-element scene? → 1/3 Rule + stagger
   patterns.

## Three pillars

Every animation must satisfy three pillars before any technical
decision:

| Pillar | Question | Drives |
|--------|----------|--------|
| **Emotional intent** | What should the viewer FEEL? | Easing, timing, amplitude |
| **Visual narrative** | What is the micro-story? | Setup → Action → Resolution |
| **Motion craft** | How do we make it believable? | Physics, secondary motion, paths |

**Three motion layers** — flat animation is missing layers:
- **Primary**: the main action the viewer follows.
- **Secondary**: supporting richness (shadows, icons shifting).
- **Ambient**: background life (gradients, subtle pulses).

## Motion personality

Select ONE archetype per project and apply it consistently.

| Archetype | Duration | Easing | Overshoot | Keywords |
|-----------|----------|--------|-----------|----------|
| **Playful** | 150–300ms | ease-out-back | 10–20% | fun, whimsical, bouncy, cute |
| **Premium** | 350–600ms | cubic-bezier(0.4,0,0.2,1) | 0% | elegant, minimal, luxury, sophisticated |
| **Corporate** | 200–400ms | cubic-bezier(0.2,0,0,1) | 0–3% | clean, professional, business, dashboard |
| **Energetic** | 100–250ms | ease-out-expo | 15–30% | dynamic, energetic, bold, exciting |

**Default**: Corporate for UI, Playful for illustrations.

**Brand motion identity** — define three constants:
1. **Signature easing**: one curve for 80% of animations.
2. **Duration palette**: three durations (quick / standard / slow).
3. **Entrance pattern**: one consistent entry style.

## Property selection

| Effect goal | Primary property | Secondary properties |
|-------------|------------------|---------------------|
| Entrance / exit | position | opacity, scale |
| Emphasis / attention | scale | rotation (subtle), opacity pulse |
| State change | opacity, color | scale (press feedback) |
| Direction / flow | position | rotation (follow path) |
| Depth / 3D feel | scale + shadow | position (parallax) |
| Loading / progress | rotation (spinner) | scale, opacity pulse |
| Success | scale (pop) | color, rotation (checkmark draw) |
| Error / alert | position (shake) | color, rotation (wobble) |

**Simplicity threshold**: use the minimum properties needed. One =
direct. Two = polished. Three or more = potentially overwhelming.

## Duration table

| Element type | Duration | Rationale |
|-------------|----------|-----------|
| Tooltip / micro-feedback | 80–120ms | Must feel instant |
| Button press / toggle | 120–180ms | Responsive feedback |
| Icon transition | 150–250ms | Clear state change |
| Card enter / exit | 200–350ms | Spatial awareness |
| Modal / dialog | 300–400ms | Focus shift |
| Page transition | 400–600ms | Context switch |
| Dramatic reveal | 600–1200ms | Theatrical build |
| Ambient | 2000–20000ms | Background life |

**Distance scales duration**: 50px = 0.8x. 100px = base. 200px =
1.3x. 300px = 1.5x. 400px = 1.6x. Full screen = 1.8–2.0x.

**Enter > exit**: entrances run 30–50% longer than exits (exit =
65–75% of entrance). Users care about what appears.

**Interactive feedback maxima**: hover <100ms; press/tap <150ms;
drag start <50ms; release/settle 200–300ms; error shake
300–400ms; long press 500–800ms.

**Duration by personality** (quick / standard / slow): Playful
150 / 250 / 400ms; Premium 350 / 500 / 800ms; Corporate 200 /
300 / 450ms; Energetic 100 / 180 / 300ms.

## Easing selection

**Directional rules**: entrance → decelerate (ease-out family,
fast start, gentle landing); exit → accelerate (ease-in family);
on-screen → smooth both ends (ease-in-out); looping ambient →
seamless (sine ease-in-out); rotation, progress bars, timers →
linear, the only place linear is allowed for spatial movement.

**Industry standards:**

| Standard | Cubic bezier | Use for |
|----------|-------------|---------|
| Material Design 3 | (0.2, 0, 0, 1) | Default on-screen |
| MD3 Emphasized | (0.05, 0.7, 0.1, 1) | Entrances, attention |
| MD3 Accelerate | (0.3, 0, 1, 1) | Exits, dismissals |
| MD3 Decelerate | (0, 0, 0, 1) | Entering |
| Apple HIG | (0.25, 0.1, 0.25, 1) | Standard iOS |
| Snappy UI | (0.2, 0, 0, 1) | Fast, decisive |
| Gentle float | (0.4, 0, 0.2, 1) | Ambient, background |
| Bounce settle | (0.175, 0.885, 0.32, 1.275) | Overshoot, playful |
| Elastic snap | (0.68, -0.55, 0.265, 1.55) | Dramatic |

**Material-based easing:**

| Material | Duration scale | Overshoot |
|----------|---------------|-----------|
| Rigid (metal, stone) | 1.2x | 0% |
| Elastic (rubber, gel) | 0.8x | 15–25% |
| Fluid (water, paint) | 1.5x | 5% |
| Paper (cards, sheets) | 1.0x | 3–5% |
| Gas (smoke, fog) | 2.0x | 0% |
| Glass (brittle) | 0.9x | 0% |

**Spring parameters** (stiffness / damping): very stiff 400+ /
25–30; standard 250–350 / 18–24; bouncy 150–250 / 10–15; very
bouncy 100–200 / 5–10; gentle 100–150 / 20–25. Damping below 1.0
oscillates, 1.0 is the fastest settle without oscillation, above
1.0 settles slowly.

## Disney's 12 principles — UI adapted

1. **Squash and stretch** — squash scale ~[1.2, 0.8], stretch
   ~[0.85, 1.15]; impact 30–65ms, recovery 65–130ms; preserve
   volume (width +20% → height falls proportionally). Skip for
   premium/luxury brands.
2. **Anticipation** — a small motion opposite to the main direction
   before the action; 100–200ms, 10–20% of the main action.
   Button: scale down 3% before expanding; card: shift 5–10px away
   first. Skip for micro-feedback under 150ms.
3. **Staging** — dim non-hero elements to 40–60% opacity (optional
   2–4px blur); hero enters 100–200ms after supporting elements;
   one primary action per timing beat.
4. **Straight ahead vs pose to pose** — straight ahead is fluid and
   spontaneous (particles, ambient, generative art); pose to pose
   is planned and controlled (UI transitions, state changes).
5. **Follow through and overlapping action** — child delay 50–150ms
   behind parent; trailing elements offset stop times by
   100–200ms; use spring easing for trailing parts (lower stiffness
   = more trailing).
6. **Slow in and slow out** — entrance ease-out, exit ease-in,
   on-screen ease-in-out, ambient loop sine ease-in-out. Never
   linear for spatial movement.
7. **Arcs** — add a 10–20px perpendicular offset at the path
   midpoint; subtle (5px) for corporate, pronounced (20px+) for
   playful. Mechanical UIs may use straight paths deliberately.
8. **Secondary action** — amplitude 30–50% of primary, timing
   50–100ms after primary, different easing than primary. Card
   enters → shadow grows; button presses → ripple expands.
9. **Timing** — heavy (modals, pages) 400–800ms; light (tooltips,
   toggles) 100–250ms; sad/serious 600ms+; happy/light 200–400ms;
   urgent 100–200ms. Entrances 30–50% longer than exits.
10. **Exaggeration** — Playful 15–25%, Energetic 20–30%,
    Corporate 0–5%, Premium 0%. Scale overshoot 10–30% beyond
    target; rotation ±5–15°.
11. **Solid drawing** — consistent proportions across keyframes;
    scale + rotation together for depth; shadow behaviour matches
    the implied light source.
12. **Appeal** — smooth curves over sharp angles, satisfying timing,
    personality consistent across all elements. Appeal killers:
    jerky motion, inconsistent timing, abrupt stops, uniform
    animation.

**Combining principles** (recipes): button press = anticipation +
squash/stretch + follow-through + secondary + timing; card entrance
= anticipation + arcs + slow in/out + follow-through + staging;
success celebration = exaggeration + secondary + timing +
squash/stretch + appeal; error shake = timing + slow in/out +
staging (no exaggeration); loading spinner = timing + slow in/out
+ appeal.

## Choreography essentials

**Coordinated entry**: lead with the hero (largest displacement,
most attention-grabbing easing); spatial origin consistency (all
elements enter from the same direction or shared origin — mixed
directions read as chaos); counter-motion (hero moves right →
ambient moves left at 20–30% speed; hero scales up → shadow scales
down 10–20%; hero lifts → shadow spreads and softens 20–30%).

**Sequence structure**: setup 20–30% (elements enter, scene
establishes), action 30–40% (primary motion, hero moment),
resolution 30–40% (settle, secondary reactions, breathing). Leave
100–200ms stillness after resolution before new motion.

**1/3 Rule (distance)**: no motion travels more than 1/3 of the
screen without a keyframe change.

**1/3 Rule (elements)**: with 3+ elements, no more than 1/3 in
active motion simultaneously.

**Stagger budgets** (delay / total budget): micro cascade 20–40ms
/ <200ms (list items, grid cells); standard 50–100ms / <400ms
(cards, panels, nav); dramatic 100–200ms / <600ms (hero sections);
wave 30–60ms / <500ms (data visualisations). Total stagger must
stay under 500ms. Direction: top-to-bottom (lists), left-to-right
(horizontal), center-out (hero), random (organic), reverse (exits).
All staggered elements use the same easing family; vary only start
time, not curve. The last element may get a slight overshoot as
punctuation.

**Shared motion events**: when several elements react to one
trigger, all start within 50ms of each other; they may arrive at
different times; same easing family; motion originates from the
trigger point.

**Depth through speed**: foreground 1.0x displacement, fastest;
midground 0.5x, medium; background 0.2x, slowest.

**Common recipes**: dashboard load — skeletons fade in (100ms),
hero metric (250ms ease-out, 100ms delay), supporting cards stagger
(50ms between, 200ms each), chart data draws in (300ms, starts with
cards), ambient pulse on the primary metric. Modal open — background
dims (200ms), modal scales 95%→100% + fades (300ms, 50ms delay),
content fades in (200ms, 100ms after modal), close button last
(150ms). List update (item added) — existing items shift down
(200ms ease-in-out), new item fades + slides from top (250ms
ease-out, 50ms delay), subtle scale overshoot on land (3–5%).

## Common patterns

**Button press (Playful)**: anticipation — scale to 0.97 (50ms
ease-out); squash — scale to [1.04, 0.96] (100ms ease-in); follow
through — overshoots to 1.02, settles to 1.0 (spring, 200ms);
secondary — shadow shrinks during press, icon shifts down 2px.
Total: ~150ms press + 200ms settle.

**Card entrance (Premium)**: start 20px below target, opacity 0;
path — slight curve (10px X offset at midpoint); easing —
ease-out-cubic deceleration; follow through — shadow arrives 50ms
after card; secondary — content fades in 100ms after card lands;
staging — other cards dim to 80%.

**Success state (Playful)**: primary — scale pop with
ease-out-back; secondary — checkmark draws in; ambient — subtle
particle burst; color — green fill. Total 300–400ms.

**Error shake (Corporate)**: primary — position oscillates 2–3
times, ±10–15px horizontal; easing — ease-in-out for sharp stops;
color — red tint. Total 300–400ms. No overshoot: errors feel firm.

## Emotion-to-motion map

| Emotion | Character | Path | Easing | Duration |
|---------|-----------|------|--------|----------|
| Joy | Bouncy, arcs | Curved, upward | ease-out-back | 200–400ms |
| Calm | Smooth, flowing | Gentle curves | sine ease-in-out | 500–1000ms |
| Urgency | Sharp, fast | Straight lines | ease-out | 100–200ms |
| Sadness | Slow, downward | Drooping curves | cubic ease-in-out | 600–1200ms |
| Surprise | Sudden, expanding | Radial outward | ease-out-expo | 150–300ms |
| Elegance | Slow, controlled | Long arcs | (0.4,0,0.2,1) | 400–700ms |
| Playfulness | Bouncy, irregular | Arcs, squiggly | ease-out-back | 200–350ms |

**Path as language**: angular = tense; curved = friendly; spiral =
whimsical; diagonal = purposeful; vertical = growth/weight;
horizontal = progress.

## Weight classification

| Weight | Examples | Duration | Overshoot | Easing |
|--------|----------|----------|-----------|--------|
| Heavy | Modals, overlays | 300–500ms | 0% | Gentle, high damping |
| Medium | Cards, panels | 200–350ms | 3–5% | Moderate |
| Light | Tooltips, badges, icons | 80–200ms | 5–15% | Responsive |

**Overshoot budget**: success 5–10%; error 0%; feedback 2–5%;
celebration 15–25%; premium 0%.

## Quality rules

**CRITICAL — never break:**
1. Never linear for spatial movement — always use easing curves
   (linear only for spinners, progress bars, timers).
2. Never opacity-only for important state changes — combine with
   position or scale.
3. Never exceed 1/3 of the screen without an intermediate keyframe.
4. Always three motion layers — primary + secondary + ambient.

**HIGH — strongly follow:**
1. Match duration to element type (see tables).
2. Use directional easing (ease-out entrance, ease-in exit).
3. Apply Disney principles (especially anticipation and
   follow-through).
4. Maintain a consistent personality across the scene.

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Looks robotic | Linear easing or no arcs | Add easing curves + arc paths |
| Feels too slow | Duration too long for element type | Check duration table, use ease-out |
| Feels cheap / flat | Missing secondary + ambient | Add shadow motion + background life |
| Too distracting | Too many elements moving | Apply 1/3 rule, reduce amplitude |
| No personality | Generic easing everywhere | Apply personality archetype consistently |

## Provenance

Ported from the LottieFiles `motion-design-skill`
(https://github.com/LottieFiles/motion-design-skill, MIT,
Copyright (c) 2025 LottieFiles). The upstream skill's 16 director /
patterns / reference files are condensed here into Devon's format;
the tables and principle numbers are carried across because they
are the content. Where Devon's substrate is named
(`keyframes.py`, `transitions.py`, her web surfaces), that is
Devon's own grounding, not upstream text.
