# Goals Sweep — External Mining

Module: `nomorals/goals/` — `GoalTracker`, `IdeaTracker`, dataclasses
`Goal/Subgoal/ProgressEntry/Reminder/Idea`, `GOAL_TEMPLATES`.

Method: for every class, find the best outside implementations (repos, docs,
papers, production tools — even rough builds hold gold), then port the gold.

## What the module already has
Durable goals w/ subgoals, progress history, streaks, templates, reminders,
deadlines, stats, markdown briefing; idea cards w/ dismiss tracking,
promotion to goals, LIKE search. Gaps found by asking "what SHOULD this have":
no measurable key results, no dependencies, no prioritization algorithm, no
pace/forecast math, no recurring nudges, no journal, no weekly review, no goal
search, no idea scoring, no custom templates, no export, no milestones.

## 1. Taskwarrior urgency algorithm → `focus_score` / `today_focus`
Source: https://taskwarrior.org/docs/urgency/ and the coefficient table at
https://github.com/timcase/taskwarrior-agent-skill/blob/HEAD/references/urgency.md
(also verified port in ccextractor/taskwarrior-flutter).

Gold: urgency is a weighted polynomial over binary/graded terms with
**configurable coefficients**; defaults:
- `+next` special tag: 15.0 (dominates everything)
- due: 12.0, value graded — ≥7 days overdue → 1.0; −14…+7 days →
  `((days_overdue + 14.0) * 0.8 / 21.0) + 0.2`; >14 days out → 0.2
- blocking: 8.0 (1.0 if it blocks others); blocked: −5.0; waiting: −3.0
- priority: H 6.0 / M 3.9 / L 1.8
- active (started): 4.0; scheduled: 5.0
- age: 2.0 × (age_days / 365); annotations: 1.0 graded 0/0.8/0.9/1.0 by count;
  tags: 1.0 graded the same; project: 1.0

Port: goal-level urgency adapted 1:1 — `next` tag → 15.0; target_date graded by
the exact due formula; priority int → H/M/L bands (≥3/2/1); progress>0 →
"started" 4.0; age term; subgoals-as-annotations; tags term; goal-links supply
blocking (+8.0) / blocked (−5.0); paused → waiting (−3.0). Staleness
(no progress in N days) gets its own positive term — no TW equivalent, but
Beeminder (below) justifies treating silence as signal. Coefficients kept in a
`FOCUS_COEFFICIENTS` dict so users can retune, exactly like TW's config.

## 2. Loop Habit Tracker (iSoron/uhabits) → `momentum`
Sources: https://github.com/iSoron/uhabits/discussions/689 ("How is the score
calculated?") and the community TS port at
https://github.com/laurenj3250-debug/fairy-bubbles/blob/HEAD/GoalConnect/docs/plans/2025-11-21-habit-score-and-flexible-frequency.md
(ported from `Score.kt`).

Gold: habit **strength** is exponential smoothing, not a raw streak —
recent repetitions weigh more than old ones, so one miss after a long streak
barely dents the score while frequent misses sink it. Formula:
`multiplier = 0.5 ** (sqrt(frequency) / 13)`; `score = prev*multiplier +
completed*(1-multiplier)`.

Port: per-user `momentum()` over the last 60 days of progress-history
checkmarks (any progress record that day = completed), frequency = 1.0
(daily). Raw streak stays (`progress_streak`); momentum is the forgiving
companion metric. Also Loop's per-habit reminders map to recurring reminders
+ snooze below.

## 3. Beeminder → `pace_report`, `check_in`
Sources: https://blog.beeminder.com/less/ (yellow brick road), PPR spec
https://doc.beeminder.com/ppr, help https://help.beeminder.com/article/157-pessimistic-presumptive-reports.

Gold:
- A goal is a **road** (current → target over time); the dashboard shows
  **safe days** = days until derail if no new data.
- Flatline on a do-more goal derails by itself; do-less goals get
  **pessimistic presumptive reports** (assume the worst) so silence never
  accrues fake buffer.

Port: `pace_report(goal_id)` computes required daily pace
`(100 − progress) / days_left`, actual pace from history slope, projected
completion date, safe-days buffer, and a verdict
`on_track | at_risk | off_track | overdue | no_target | no_data`.
`check_in()` is the lightweight datapoint: records a history entry (keeps the
streak alive) without forcing a progress change — Beeminder's "enter a 0"
for do-less goals.

## 4. OKRs → Key Results
Sources: Google's OKR practice via https://github.com/src-d/okrs/blob/master/sourced-okr-methodology.md
(Perdoo-sourced: "a KR is a metric with a starting value and a target value")
and the OKR template at
https://github.com/product-on-purpose/product-lifecycle-templates/blob/HEAD/templates/okrs/okrs_template-full.md
(baseline is not optional; mark committed vs aspirational up front).

Gold: 2–5 **measurable** key results per objective, each with
**baseline → target** (not just a checkbox), graded 0.0–1.0; KRs describe
**outcomes, not activities**; milestone KRs are legitimate for phased work.

Port: `goal_key_results` table — title, unit, baseline, target, current,
weight, direction (increase/decrease), committed flag. `okr_score()` =
weight-averaged 0–1 grade; goal progress auto-rolls up from KRs when present
(subgoals otherwise, manual otherwise — precedence documented). Milestones
table covers the "milestone KR" case with dated checkpoints.

## 5. Habitica → goal kinds, checklists, reordering, daily cadence
Sources: https://habitica.fandom.com/wiki/Establishing_Your_Tasks and
https://habitica.fandom.com/wiki/Task_Type_Choice:_Habit,_Daily,_or_To_Do;
clone https://github.com/romantokar/my-habitica (four task types, streaks,
checklists, drag-and-drop reorder).

Gold: Habits (±, no schedule), Dailies (scheduled, **streaks**, missed-day
penalties), To-Dos (one-time, **checklists**), Rewards. Checklists live inside
tasks; reorder is drag-and-drop.

Port: goals get a `kind` (`outcome` | `habit` | `project` — Habitica's
habit/daily/todo trichotomy mapped onto goals); subgoals are the checklist
layer and gain `rename_subgoal` + `reorder_subgoals` (drag-and-drop analog);
habit-kind goals pair with recurring daily reminders.

## 6. GTD weekly review → `weekly_review`
Sources: David Allen's 11-step checklist via
https://gettingthingsdone.com/2018/08/episode-43-the-power-of-the-gtd-weekly-review/
(GET CLEAR / GET CURRENT / GET CREATIVE) and the operationalized checklist at
https://github.com/okayiris/registry/blob/HEAD/skills/daily-and-weekly-review/SKILL.md.

Gold: Get Clear (inbox → zero, empty head) → Get Current (next actions,
past/future calendar, **waiting-fors, every project needs ≥1 next action**,
checklists) → Get Creative (**review someday/maybe**, capture new ideas).

Port: `weekly_review(user_id)` assembles exactly that: completed this week,
overdue, due soon, **stale goals** (no activity ≥7d — the "no next action"
smell), blocked goals, ideas awaiting review (the someday/maybe list), and
suggested actions. `stale_goals()` is the primitive; briefing gains a review
section.

## 7. ICE scoring (Sean Ellis) → `score_idea` / `top_ideas`
Sources: https://github.com/hu9osaez/agents/blob/HEAD/plugins/ai-pm-copilot/skills/prioritization-methods/SKILL.md
and https://github.com/melgarafael/growthos/blob/HEAD/skills/marketing-strategy/SKILL.md.

Gold: ICE = (Impact + Confidence + Ease) / 3, each 1–10, built for fast
triage of idea backlogs; documented pitfalls (confidence inflation, ease bias)
kept as docstring guidance.

Port: ideas gain optional impact/confidence/ease + computed ICE; `top_ideas()`
ranks active ideas by ICE; unscored ideas sort last. `find_similar_ideas()`
(Jaccard token overlap) is the dedupe companion — even rough builds had one.

## 8. Supporting gold
- **Loop/CSV export** (uhabits exports CSV+SQLite) → `export_json` /
  `export_markdown` for the whole goal system.
- **Mindwtr** (https://github.com/dongdongbh/Mindwtr) local-first GTD:
  "come back after two weeks away and get a manageable next step, not a guilt
  trip" → stale-goal handling nudges instead of shaming; weekly review is
  guided, not a wall of red.
- **Strides-style goal types** (target/habit/milestone/project) → `kind`
  field + `pace_report` (target) + recurring reminders (habit) + milestones.
- **super-productivity / taskwarrior recurrence** → recurring reminders
  (daily/weekly) that auto-reschedule on ack; `snooze_reminder`.

## What was deliberately NOT ported
- Gamification currencies/XP (Habitica gold/HP): no economy exists in this
  repo; would be decoration.
- Beeminder monetary stakes: out of scope for a tracker library.
- Real full-text search (FTS5): LIKE search matches the existing idea search
  and keeps the offline/sqlite-simple contract.
- Multi-user sharing/collaboration: single-user design stands.
