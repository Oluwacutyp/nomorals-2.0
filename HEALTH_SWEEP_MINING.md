# HEALTH SWEEP — External Mining Report

Mined 2026-10-10 for the `nomorals/health/` module sweep (9 functional
files + `__init__.py`). For every significant class: how does the
best implementation of X do it — and what gold did we take.

Sources: WHOOP (podcast eps 040/084, Healthcare Discovery 2026 review),
Oura (white paper v3, support docs), Gilbert (noop-public-release
`RECOVERY_EFFORT_REST_FACTORS.md`), Ada Health (JMIR safety studies,
EU-MDR IIa), NHS-adjacent triage research (BMC Health Services Research
2025), Ochy (ochy.io), MediaPipe Pose (33 landmarks), JuggernautAI /
RP Hypertrophy / Alpha Progression / MacroFactor (sensai.fit 2026
comparison), Strava gamification analysis, NIH/NIDDK photo-calorie study
(Hengist, NUTRITION 2026), Daylio, Bearable, Chronic Insights,
over-training sports-science literature.

---

## 1. coach.py — HealthCoach / readiness (ask-your-data coaching)

**How the best do it:**

- **WHOOP Recovery** = 3 inputs, HRV-dominant: HRV vs personal baseline
  **0.55**, resting HR vs baseline **0.20**, sleep performance **0.15**,
  respiratory rate **0.05**, skin-temp deviation **0.05**. HRV "carries
  most of the predictive value"; RHR/sleep matter most when they
  *diverge* from HRV. Bands: ≥67 green, 34–66 yellow, <34 red.
- **Gilbert** (open-source re-implementation): `z = Σ(termZ ×
  weight) / Σ(weight)` — **missing terms drop out and renormalize**,
  then a logistic squash anchored so z=0 → ~52% ("honesty floor").
  Rest/sleep = 0.45·duration + 0.20·efficiency + 0.25·restorative +
  0.10·consistency, with restorative = (deep+REM)/asleep vs 50% target.
- **Oura Readiness** = 9 contributors vs **personal 14–30-day baselines**
  (never population averages); Resilience = weighted average of the
  last 14 days, **minimum 5 days of data per contributor**, more weight
  on recent days. Takes ~2 weeks to learn your averages.
- **WHOOP Sleep Need is DYNAMIC** — shaped by recent strain, sleep
  debt, naps, circadian rhythm. An athlete who slept 7h but needed 8h
  scores differently than one who needed 7h. Ours uses a fixed 7–9h
  window.

**Gold taken:**

1. Renormalizing weights — missing signals redistribute instead of
   dragging the score down (already partially done; made explicit and
   WHOOP-weighted).
2. **Dynamic sleep need** — `sleep_need()`: base 8h, +debt repayment,
   +strain premium, −nap credit. Readiness compares last night to
   *your need*, not a fixed band.
3. **Contributor breakdown** — `explain_readiness()`: Oura-style "why"
   (which input moved the score, vs personal baseline).
4. Baseline honesty: HRV score needs ≥7 days of history before it
   carries full weight (Oura's 2-week learning, Gilbert's null-without-
   baseline).
5. Band thresholds aligned to WHOOP (67/34) with per-band coaching.
6. `readiness_history()` — score over time for trend display.

## 2. training.py — TrainingCoach (plans + readiness gating)

**How the best do it:**

- **JuggernautAI**: individualized volume landmarks (enough to grow,
  not so much you can't recover), optimized frequency, weak-point
  exercise selection, **real-time load adjustments from per-set RPE
  feedback**, 0–100 readiness, habit tracker, 250+ exercise DB.
- **RP Hypertrophy**: deloads scheduled automatically at end of each
  block; weekly sets auto-managed.
- **Alpha Progression**: set-by-set weight/rep targets, not just plans.
- **Progression science**: double progression — add reps each week
  (8→9→10), then add weight, drop reps, rebuild.

**Gold taken:**

1. **Volume landmarks** — `weekly_sets()` per muscle group vs
   MEV/MRV-style bands (too little / sweet spot / too much), surfaced
   on `/train volume`.
2. **Auto-deload detection** — when the last 3 weeks show rising volume
   + readiness declining, the plan suggests a deload week (RP pattern).
3. **Automatic PR tracking** — `personal_records()`: heaviest load per
   exercise key + longest streak, parsed from the workout log
   (Strava-style automatic PR detection).
4. **RPE targets on sets** — `Exercise.rpe_target`, and
   auto-regulation: RPE ≥9 two sessions running → hold progression;
   RPE ≤6 → double bump (Juggernaut auto-regulation loop).
5. **Weak-point mapping** — form-analysis issues flow into exercise
   selection notes ("add face pulls — upper-back flagged twice").

## 3. challenges.py — ChallengeStore (photo-proof, streaks, squads)

**How the best do it:**

- **Strava**: challenges auto-track progress from uploaded activities;
  **automatic PR tracking** (parses activities for benchmark
  distances); **personal medals** — your own segment PBs earn
  1st/2nd/3rd medals so non-elites still win; segmented leaderboards
  (age/weight/club filters) keep competition fair; monthly challenges
  with permanent profile badges.
- **Duolingo**: streak freeze (earned safety net), loss-aversion
  nudges, streak repair.

**Gold taken:**

1. **Streak freeze** — `earn_freeze()` / auto-applied on miss; freezes
   earned per 7-day streak (Duolingo pattern).
2. **Recurring challenge templates** — `monthly_challenge()` factory
   ("October Distance Challenge") with permanent badge (Strava).
3. **Personal medals** — per-member PB tracking inside a challenge:
   first time hitting your own best → 🥇/🥈/🥉 in the leaderboard.
4. **Auto-progress from training logs** — `sync_from_training()`:
   TrainingCoach workout_log entries count toward active
   workout-count challenges automatically (Strava auto-tracking).
5. Progress-bar formatting on `/challenge board`.

## 4. form.py — form coach (video movement analysis)

**How the best do it:**

- **Ochy**: proprietary pose estimation → joint angles → gait events
  (footstrike, toe-off), cadence, ground contact time, flight time →
  **corrective algorithm** mapping measurements to targeted exercises.
- **MediaPipe Pose** (open, CPU-friendly): 33 body landmarks per
  frame → normalize (hip-centered, torso-scale) → joint-angle features
  → geometric rules / classifiers. Used by multiple open-source gym
  form correctors with 95% video-level accuracy claims.
- Skill-file pattern: keep raw observation / derived metric / model
  inference / confidence **distinct**; state visibility limitations;
  stop when landmarks are unreliable.

**Gold taken:**

1. **Real quantitative path** — `estimate_pose_frames()` via MediaPipe
   (import-guarded; dep absent → falls back to the Seer seam, never
   fabricates). Fills the `ANGLE_HOOKS` (knee/hip/spine/ankle) with
   measured degrees.
2. **Rule-based angle checks** — `angle_rules()` per movement
   (e.g. squat: knee angle at bottom 80–130°, spine lean <45°) → PASS/
   FLAG with measured numbers, merged with the Seer qualitative read.
3. **Rep counting** — joint-angle oscillation counting from frame
   sequences (`count_reps()`), surfaced in the analysis.
4. **Tempo read** — eccentric/concentric phase timing estimate from
   frame cadence.
5. `form_trend()` — score history per exercise from FormStore
   (progress over time, the retention hook).

## 5. previsit.py — triage routing / navigation

**How the best do it:**

- **Ada Health**: 4-level urgency (emergent/urgent/routine/self-care —
  same taxonomy as ours), **17–40 clarifying questions per consult**
  to surface red flags the user didn't volunteer; conservative
  overtriage accepted as the safe direction (94.7% safe in the ED
  study); care navigation + clinical handover reports; EU-MDR Class
  IIa certified.
- BMC 2025: online checkers ask **far fewer red-flag questions than
  PCPs (36.9% vs 71.8%)** — the gap is follow-up questions, and LLMs
  can close it with free-text input.

**Gold taken:**

1. **Clarifying-question flow** — `clarify_questions()`: after an
   initial route, targeted follow-ups ("any chest pain with it?",
   "how long?") re-run the router; conservative bump-up preserved.
2. **Timeline correlation** — `route_with_context()`: prior similar
   episodes from the health timeline ("3rd headache log this month")
   adjust urgency.
3. **Handover report** — `handover_report()`: printable care-navigation
   packet (symptoms, timeline, meds, questions) for the doctor — Ada's
   clinical handover.
4. Route confidence + explanation (`Route.confidence`, why each level).

## 6. nutrition.py — meal logging

**How the best do it:**

- **NIH/NIDDK 2026** (Hengist, NUTRITION 2026, n=102 metabolic-kitchen
  meals): MyFitnessPal, LoseIt!, CalAI, Appediet ALL underestimated
  **250–345 kcal and ~30g fat per meal (~33%)**; worse on high-fat/
  keto meals; carbs estimated best, fat worst. Researchers recommend
  **combining AI photo with traditional tracking** — exactly our
  2-question design. Validates it; we now cite the study year.
- Macro tracking is standard everywhere; Nigerian food DBs are a
  differentiator nobody else has.

**Gold taken:**

1. **Macro estimates** — protein/carbs/fat ranges added to every
   `FoodEntry` (Nigerian dishes, estimated from standard references;
   marked as estimates).
2. **Daily totals** — `daily_totals()`: today's logged meals →
   calories + macros vs targets, formatted as a day card.
3. **Water tracking** — `log_water()` / hydration target (Daylio-style
   factor), included in daily card.
4. **Repeat-meal shortcut** — `repeat_last()` / "same as yesterday"
   (the highest-frequency real-world action).
5. Meal history query (`meals_on()`).

## 7. patterns.py — mood-pattern detection

**How the best do it:**

- **Daylio**: activity↔mood correlation charts, weekly/monthly/yearly
  mood stats, best/worst days, year-in-pixels mood map.
- **Bearable**: factor tracking across meds, supplements, foods,
  activities, symptoms → "connecting subtle dots between myriad
  tracked factors"; custom dashboards, data export.
- Our module only correlates 3 factors (sleep, activity, caffeine).

**Gold taken:**

1. **More factors** — medication, supplement, social, screen-time,
   weather-manual, exercise-intensity (Bearable pattern); medication
   and sleep-quality factors mined from timeline text.
2. **`mood_stats()`** — weekly/monthly aggregates: avg mood, low-day
   count, best/worst day (Daylio pattern).
3. **`mood_map()`** — year-in-pixels style ASCII mood calendar.
4. **Lag analysis** — n-day lag correlations ("good mood tends to
   follow active days by 1 day").
5. `best_worst_days()` — the Daylio retention view.

## 8. drift.py — drift detection + recovery

**How the best do it:**

- **Overtraining sports science**: verdicts need **≥2 weeks** of
  measurements, not days; signals = elevated RHR, HRV drop, mood
  changes, sleep disturbance, frequent illness, performance decline.
  Ours is 3-day — good for consumer early-warning, but a second
  slower window catches real drift.
- **Oura Resilience**: weighted average of the last 14 days, min 5
  days of data, recent days weighted more.
- **WHOOP**: respiratory-rate deviation and illness flagging from
  RHR/temp/HRV moving together.

**Gold taken:**

1. **Two-window detection** — acute (3-day, existing) + chronic
   (14-day, Oura-resilience-weighted); chronic signals get their own
   kinds (`chronic_sleep_debt`, `chronic_mood_slide`).
2. **Illness early-warning** — `illness_watch()`: RHR elevation +
   HRV drop + sleep disturbance moving together → gentle
   "something may be coming on" (WHOOP pattern), with crisis
   resources when mood is involved.
3. **Recovery trajectory** — `projected_recovery()`: at the current
   slope, estimated nights to return to baseline (forecast, honest
   band).
4. Escalation: 3+ consecutive "act" days → stronger guidance +
   suggestion to talk to a professional.

## 9. timeline.py — health timeline (tracking)

**How the best do it:**

- **Chronic Insights**: unlimited vitals with charts, medication
  tracking, factor tracking, PDF export, custom analysis charts,
  weather correlation.
- **HealthDiary**: centralized records, vital-sign charts, appointment
  reminders, PDF/Excel export, photo attachments.
- **Symptom-tracker apps**: color-coded severity sliders, trigger
  chips, doctor-report tab ("walk in with a month of facts").

**Gold taken:**

1. **Vitals trends** — `vitals_trend()`: parse measurement events
   (BP "120/80", weight "72kg") into series + ASCII sparkline charts.
2. **Doctor report** — `export_report()`: markdown handover
   (timeline + meds + vitals trends), the "walk in with facts" view.
3. **Symptom stats** — `symptom_stats()`: per-symptom frequency,
   avg severity, trend arrow (the trigger-trend math).
4. **Full-text search** — `search()` across event text.
5. **Logging streaks** — `logging_streak()` (retention mechanic,
   shared pattern with challenges).

## 10. Cross-cutting style upgrades

- Readiness/training/challenge outputs get consistent card headers,
  progress bars (████░░ 60%), band emojis, and "why" sections.
- Every numeric claim keeps its honest-data framing (no fabricated
  numbers — the module's core virtue, preserved everywhere).
- New outputs pass the existing banned-phrase guards
  (`guard_coaching`, `_check_banned`, `_assert_safe`) — test-enforced.
