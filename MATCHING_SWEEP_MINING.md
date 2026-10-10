# MATCHING Sweep — External Mining Report

Module: `nomorals/matching/` (5 files: `__init__`, `stable`, `batches`,
`questionnaire`, `chat`). Date: 2026-10-10. Written BEFORE any
implementation, per sweep method.

Every class was compared against the best implementations outside this repo.
"Best" = mined for gold to merge. Each section ends with the gaps this sweep
will fill. All URLs below were returned verbatim by web search.

---

## 1. `stable.py` — `stable_match` / `verify_stable` (Gale–Shapley)

### How the best do it

**Hinge "Most Compatible" (production Gale–Shapley at scale).**
Hinge pairs members using "a combination of machine learning and the
Nobel-prize-winning Gale Shapley algorithm" — ML learns each user's taste
profile from like/pass activity, then deferred acceptance pairs people who
are likely to prefer each other *mutually*. The pick refreshes every 24
hours and both parties see each other simultaneously. In trials the feature
was **8x more likely to result in dates** than baseline discovery.
Sources: https://www.bustle.com/wellness/how-does-hinge-algorithm-work,
https://techcrunch.com/2018/07/11/hinge-employs-new-algorithm-to-find-your-most-compatible-match-for-you/,
https://thehustle.co/hinge-machine-learning-algorithm

**daffidwilde/matching (JOSS paper, MIT, the reference Python library).**
Four game types: `StableMarriage`, `HospitalResident` (with **capacities**),
`StudentAllocation`, `StableRoommates`. API shape:
`StableMarriage.create_from_dictionaries(suitor_prefs, reviewer_prefs)` then
`game.solve()` → dict-like `SingleMatching`. HR takes a `capacities` dict.
Source: https://github.com/daffidwilde/matching

**Market-design literature (npbuilds/skill-library matching-markets SKILL).**
Deferred Acceptance: stable, strategy-proof for the proposing side,
proposer-optimal. Top Trading Cycles (Shapley & Scarf 1974): strategy-proof
for ALL participants and Pareto efficient, but NOT stable. Serial
Dictatorship / RSD: strategy-proof, Pareto efficient, trivially simple.
Every practical mechanism sacrifices at least one property — the
**stability-vs-efficiency tradeoff** is the central design decision.
Source: https://github.com/npbuilds/skill-library/blob/HEAD/./skills/game-theory/mechanism-design/matching-markets/SKILL.md

### The gold we lack

1. **Capacities (Hospital–Resident).** Ours is 1:1 only. The module's own
   docstring promises "mentor ↔ mentee" (#46) — a mentor takes *several*
   mentees. daffidwilde's `HospitalResident` is the API reference. We need
   resident-proposing deferred acceptance with per-reviewer quotas.
2. **Single-pool matching (Stable Roommates, Irving 1985).** The module
   promises "member ↔ member" community matching — one pool, no two sides.
   Irving's two-phase algorithm (proposals → rotation elimination) finds a
   stable pairing when one exists and *honestly reports* when none exists
   (unlike SM, SR instances can be unsolvable — Gale & Shapley's original
   1962 paper already gives a 4-person counterexample).
   Sources: https://www.cs.bu.edu/faculty/homer/537/talks/Oxana-nov12-StableRoommates.pdf,
   https://github.com/gfornari/stable-roommates-problem
3. **Allocation mechanisms.** TTC for indivisible goods (who gets which
   gig / which learning slot): each agent owns an endowment, points to the
   owner of their top choice, cycles execute. Serial dictatorship as the
   dead-simple fallback. Both are standard market-design tools we lack.
   Sources: http://mpra.ub.uni-muenchen.de/41366/1/MPRA_paper_41366.pdf,
   https://www.dcs.gla.ac.uk/~davidm/pubs/8043.pdf
4. **One-sided optimal assignment (Hungarian / Kuhn–Munkres).** When only
   scores exist (no two-sided preferences), the max-total assignment is the
   right tool — the efficiency extreme of the stability/efficiency tradeoff.
   Pure-Python O(n³) keeps the module's zero-mandatory-dependency rule.
5. **Learned preferences (Hinge's ML half).** Hinge doesn't ask for rankings;
   it *learns* them from like/pass activity. Our `rank` command is manual
   only. The batch-feedback loop (see §2) is our version of this gold.

**Gaps to fill:** capacities, Irving roommates, TTC, serial dictatorship,
Hungarian assignment, `MatchResult` rosters for the capacity case.

---

## 2. `batches.py` — `curate_daily` / `DailyBatchStore` (scarcity-as-ritual)

### How the best do it

**Coffee Meets Bagel (the scarcity-as-ritual reference product).** Small
daily batch ("bagels") delivered at noon — men up to 21, women up to 5
("Ladies Choice": women's bagels are pre-filtered to men who already liked
them). A **Discover** section shows out-of-type profiles for the
adventurous (exploration!). Mutual-like chats **expire in 7–8 days** —
artificial urgency against ghosting/pen-pal syndrome. The whole design
fights the paradox of choice: fewer, more intentional views.
Sources: https://www.datingadvice.com/online-dating/coffee-meets-bagel-review?utm_source=google&utm_medium=organic&utm_term=AMP&utm_content=%2Fonline-dating%2Fcoffee-meets-bagel-review-how-it-works&lander=https%3A%2F%2Fwww.datingadvice.com%2Fonline-dating%2Fcoffee-meets-bagel-review-how-it-works,
https://medium.com/@chnomanseo/how-niche-dating-apps-like-coffee-meets-bagels-in-2025-25fa5e48e17c

**Maximal Marginal Relevance (Carbonell & Goldstein, SIGIR 1998).**
The standard diversity re-ranker: greedily pick
`argmax λ·Sim(d,Q) − (1−λ)·max Sim(d, selected)`. λ=1 is pure relevance,
λ=0 pure diversity, 0.5–0.7 the typical balance. Used by LangChain
retrievers and RAG pipelines everywhere to stop near-duplicate stacking.
Sources: https://github.com/asg017/sqlite-vec/issues/266,
https://github.com/stasieniec/university-vault-site/blob/HEAD/content/Concepts/Maximal%20Marginal%20Relevance%20(MMR).md

**Explore/exploit (bandits) + cold start.** Production recommenders reserve
exploration budget for uncertain items (ε-greedy, UCB1, Thompson sampling
over Beta-Bernoulli like/pass feedback) and boost brand-new items past the
cold start — e.g. the Yelp hybrid recommender mined uses MMR *plus*
cold-start handling. Our batches currently never learn: no feedback is
recorded at all.
Source: https://github.com/yingwana/restaurant-recommender

### The gold we lack

1. **Diversity.** Pure relevance sorting stacks near-duplicates (five
   "python gig" picks). MMR with tag-Jaccard similarity, λ≈0.7 default.
2. **A feedback loop.** No like/pass recording → no learning. Add
   `record_feedback` (impressions + likes) and count a shown batch as
   impressions (that's what "shown" means).
3. **Exploration slots.** Thompson sampling over per-candidate
   Beta(likes+1, passes+1): under-sampled and new candidates get explored;
   winners get exploited. Seeded RNG for deterministic tests.
4. **Cold-start boost.** New candidates (zero impressions) must surface,
   not languish at the bottom forever.
5. **"Why this pick" explanations.** Every real product explains its
   ranking ("you both like X"). We show titles only.
6. **No removal path.** Candidates can be added but never removed.

**Gaps to fill:** MMR re-rank in `curate_daily`, feedback table + like/pass,
Thompson exploration slots, cold-start handling, `explain_pick`,
`remove_candidate`, batch stats.

---

## 3. `questionnaire.py` — `Questionnaire` / `DealBreaker` (the OkCupid pattern)

### How the best do it

**OkCupid's real algorithm (Rudder's TED-Ed lesson, widely reproduced).**
Three inputs per question: (1) my answer, (2) the answers I'd *accept* from
a match, (3) importance: irrelevant / a little / somewhat / very /
mandatory → weights **0 / 1 / 10 / 50 / 250**. For each question both
answered: my points += my weight if their answer is acceptable to me.
Match% = `sqrt((my_points/my_max) × (their_points/their_max)) × 100`.
Enemy% is the same with enemy answers. Only jointly-answered questions
count. The three-input shape (mine / acceptable / importance) is the whole
trick — our questionnaire only captures importance, so it can't do any of
this.
Sources: http://www.fastcompany.com/1812010/whos-telling-you-truth-about-dating-algorithms,
https://www.globaldatinginsights.com/news/okcupid-founder-explains-compatibility-algorithm-for-ted-ed/,
http://youtube.com/watch?v=m9PiPlRuy6E

**Feeld "Reflections" (the deal-breaker pattern, done right).** 165 prompts
across three sections — **Desires** (kinks, interests), **Boundaries**
(red flags, deal-breakers, safety), **Relationships** (attachment style,
preferences) — producing percentage scores per section. Desires are
captured *before you ever see a profile*; boundaries are hard filters, not
weights. Our module already front-loads deal-breakers (the Feeld pattern)
but has no sections and no desires/boundaries split.
Sources: https://www.globaldatinginsights.com/featured/feeld-releases-state-of-reflections-report-on-dating-norms/,
https://www.fastcompany.com/91508852/feeld-dating-app-sex-kinks-more-common-than-you-think-free-tool-lets-anybody-explore-taboos-reflections,
https://mantelligence.com/dating-app-onboarding-comparison/

### The gold we lack

1. **Acceptable-answer sets.** The missing second OkCupid input. Without it
   our "score" is a truthy heuristic. Add `set_acceptable()` + persist it.
2. **The 1/10/50/250 importance ladder.** Exponential weights are what make
   "very important" actually dominate. Map our 1–5 onto it.
3. **Two-sided match %.** `match_percent(other_questionnaire)` with the
   real sqrt-of-products formula — this is the member↔member and
   mentor↔mentee compatibility score the module's docstring implies.
4. **Per-question breakdown** ("you matched on 3/5, missed on budget") —
   the explainability half of every real matcher.
5. **Sections (Feeld).** desires / boundaries / practical grouping for
   starter questions; boundaries section feeds deal-breakers naturally.
6. **Mandatory ⇔ deal-breaker bridge.** OkCupid's "mandatory" is weight
   250; Feeld's boundaries are hard filters. Importance-5 on a boundaries
   question should be one tap away from a hard filter.

**Gaps to fill:** acceptable sets, weight ladder, one-sided satisfaction,
two-sided match %, breakdown, sections, capacity storage for §1.

---

## 4. `chat.py` — `control_match` (/match UX)

### How the best do it

- **CMB:** like / pass on every bagel (the feedback that trains the
  ranker); Discover for exploration; expiring chats.
- **Hinge:** deal-breaker preferences filter *before* ranking; prompts
  explain compatibility; "most compatible" is a daily ritual, not a feed.
- **Feeld:** desires declared before seeing anyone; skip-without-judgment
  (pass ≠ permanent rejection).

### The gold we lack

1. **like/pass commands** feeding the bandit loop (§2).
2. **`why` explanations** for picks and scores (§2, §3).
3. **Capacity-aware `run`** (mentor takes N mentees) and new mechanisms
   surfaced: `/match roommates`, `/match assign` (with an honest
   stable-vs-optimal label), `/match ttc`.
4. **Stability proof on demand** — we have `verify_stable` but no way to
   invoke it from chat.

**Gaps to fill:** like/pass/why/acceptable/capacity/roommates/assign/ttc
commands, capacity-aware run, better batch/match formatting.

---

## What stays

- Gale–Shapley core + `verify_stable` (correct, tested shape; Hinge-grade
  mechanism, keep).
- Batch stability (same picks all day), 7-day no-repeat with
  least-recently-shown recycling (the ritual mechanics are right).
- Deal-breakers-before-scoring (the Feeld pattern, already correct).
- Never-raises + SQLite + offline-testable (module contract, keep).
- Zero mandatory dependencies (user's standing order — Hungarian and
  Thompson sampling are implemented in pure Python / stdlib `random`).

## What's still weak (after this sweep, honestly)

- No learned preference inference: Hinge learns rankings from behavior; we
  still need explicit `rank` commands or questionnaire answers. The bandit
  loop learns *candidate quality*, not *user taste*.
- Ties in preferences are first-listed-wins (no weak/strong stability
  variants; SRT with ties is NP-complete — correctly out of scope).
- No ELO/desirability dynamics (Tinder retired Elo; correctly skipped).
- Irving needs even pools; odd-one-out sits out by lowest approval.
- Roommate/TTC/assign are one-shot commands, not a persistent market that
  re-clears daily (Hinge re-runs Most Compatible every 24h — a future
  scheduler hook, not this sweep).
