# LEARN sweep — mining findings

Module: `nomorals/learn/` — tutor.py (SocraticEngine, MasteryModel, TutorSession),
flashcards.py (MistakeNotebook), dialect.py (EkitiTutor), curriculum.py
(WAEC/JAMB syllabus + course generation).

Every technique below is verified against a real source. No invented methods.

---

## 1. Mastery modeling — Bayesian Knowledge Tracing (BKT)

**Source:** Corbett & Anderson 1995 ("Knowledge tracing: Modeling the
acquisition of procedural knowledge"); pyBKT reference implementation
(https://github.com/CAHLR/pyBKT) and its paper (https://arxiv.org/pdf/2105.00385);
individualized-BKT writeup (http://www.cs.cmu.edu/~ggordon/yudelson-koedinger-gordon-individualized-bayesian-knowledge-tracing.pdf).

The model has exactly four parameters per skill:
- `P(L0)` — prior probability the student already knows the skill
- `P(T)` — probability of transitioning not-known → known after one opportunity
- `P(G)` — guess: correct answer despite not knowing
- `P(S)` — slip: wrong answer despite knowing
- (+ optional `P(F)` forget extension: known → not-known, from pyBKT's BKT+forget)

Update equations (from the pyBKT paper):
- `P(L_t | correct) = P(L_t)(1-P(S)) / (P(L_t)(1-P(S)) + (1-P(L_t))P(G))`
- `P(L_t | wrong)   = P(L_t)P(S)     / (P(L_t)P(S)     + (1-P(L_t))(1-P(G)))`
- `P(L_{t+1}) = P(L_t|obs) + (1 - P(L_t|obs))·P(T)` (then apply forget)
- Predicted correctness: `P(C) = P(L)(1-P(S)) + (1-P(L))P(G)`

**Gold mined:** the current `MasteryModel.update()` is a linear ±delta heuristic
with no notion of guessing/slipping. BKT replaces it with a principled posterior
that handles "got it right by luck" and "knew it but slipped". The forget
extension (`P(F)`) plus time-decay handles the forgetting curve that the current
model ignores entirely.

## 2. Spaced repetition — FSRS-6 (Free Spaced Repetition Scheduler)

**Source:** open-spaced-repetition/py-fsrs (MIT; FSRS-6, 21 weights, Ye et al.
2024) as documented in ports that replicate it exactly
(https://github.com/patpateephangern/languagerevise) and the FSRS-6 spec
(https://github.com/doctorkishor/autoanki/blob/HEAD/REQUIREMENTS_SPEC.md);
algorithm explainer (https://github.com/riso19/openmedq/blob/HEAD/content/blog/spaced-repetition-science-for-medical-students.md).

The DSR model tracks per card:
- `D` difficulty 1–10, `S` stability (days until recall probability falls to 90%),
  `R` retrievability = current recall probability.
- Forgetting curve: `R(t,S) = (1 + FACTOR·t/S)^DECAY`, with `DECAY = -w20`,
  `FACTOR = 0.9^(1/DECAY) − 1` (so `R = 0.9` exactly when `t = S`).
- Next interval (inverse of the curve): `I = (S/FACTOR)·(R_target^(1/DECAY) − 1)`,
  target retention default 0.9.
- Initial values: `S0(G) = w[G−1]` for grades 1..4, `D0(G) = w4 − e^(w5·(G−1)) + 1`.
- Difficulty update with linear damping and mean-reversion; stability grows on
  recall (with hard-penalty `w15` and easy-bonus `w16` multipliers) and shrinks on
  lapse (`min(w11·D^(−w12)·((S+1)^w13 − 1)·e^(w14·(1−R)), S/e^(w17·w18))`).
- Card states in the wild: New / Learning / Relearning / Young (I<21d) / Mature
  (I≥21d) / Suspended / Buried (autoanki spec).

**Gold mined:** the current notebook delegates to an external
`RepetitionScheduler` (SM-2 era, 1987) and has no scheduling of its own. An
FSRS-lite DSR core (documented FSRS-6 update rules, minus the short-term
learning-step machine and interval fuzzing — both flagged honestly) gives the
notebook native adaptive scheduling: per-card stability/difficulty, due
prioritization by retrievability, lapse handling, state breakdowns.

## 3. Concept half-lives — Duolingo Half-Life Regression (HLR)

**Source:** Settles & Meeder, "A Trainable Spaced Repetition Model for Language
Learning", ACL 2016 (https://github.com/duolingo/halflife-regression, MIT):
`p(recall) = 2^(−Δ/h)` where `h` is the item's half-life in memory; practice
updates `h`. Reduced prediction error 45%+ vs baselines on Duolingo data; +12%
daily engagement in production.

**Gold mined:** HLR is the clean complement to BKT — BKT answers "does the
student know it", HLR answers "will they still know it tomorrow". One exponential
per skill gives the tutor principled review timing and due-card ranking without
the complexity of per-card FSRS.

## 4. Socratic dialogue — AutoTutor's EMT cycle

**Source:** Graesser et al., AutoTutor papers (https://link.springer.com/article/10.1007/s40593-015-0086-4;
https://telearn.hal.science/hal-00197320/document; AI Magazine piece at
http://faculty.tamuc.edu/dharter/pubs/journal/2001/aimag01/AIMag22-04-005.pdf).

EMT = expectation & misconception-tailored dialogue:
- Every hard question carries **expectations** (anticipated good answers/steps)
  and **misconceptions** (anticipated bugs, incorrect beliefs).
- Student turns are matched against both via semantic pattern matching.
- Dialog moves after each turn: short feedback (positive/neutral/negative) →
  **pump** ("what else?") → **hint** → **prompt** (fill-in-the-blank for a
  specific missing word) → **assertion** (tutor splices in the correct info
  after repeated failure) → summary.

**Gold mined:** the current engine has nudge/backtrack/probe but no expectation
coverage tracking, no misconception matching, and no pump→hint→prompt→assertion
ladder. This is the single biggest upgrade: a `QuestionScript` (expectations +
misconceptions) driving a real EMT move selector, with a built-in misconception
bank seeded from WAEC Chief Examiners' Reports (below).

## 5. Tutor pedagogy rubric — LearnLM

**Source:** Google DeepMind LearnLM technical report
(https://blog.google/products-and-platforms/products/education/google-learnlm-gemini-generative-ai/;
arxiv https://arxiv.org/pdf/2412.16429v2). Five principles with concrete rubric
items: inspire active learning (don't give answers away too quickly, ask
questions), manage cognitive load (appropriate length, bullet chunks, logical
order, no repetition/contradiction), adapt to the learner (affect-aware:
respond to frustration), stimulate curiosity, deepen metacognition (acknowledge
correctness, communicate a clear plan, guide mistake discovery, constructive
feedback).

**Gold mined:** applies directly to (a) the SOCRATIC_SYSTEM prompt (rewrite
around the five principles), (b) frustration detection → easier sub-question,
(c) session `report()` for metacognition, (d) `plan()` stating the session
objective up front.

## 6. Yoruba tone linguistics — fixing a real error

**Source:** Shittu (BUCLD, http://www.bu.edu/bucld/files/2015/06/Shittu.pdf),
citing Ward 1952, Bamgbose 1966b, Akinlabi & Liberman 2000: Yoruba has three
lexical tones H/M/L; the canonical **igba quintuple**:
- MH 'calabash' (= orthographic **igbá**)
- LL 'time' (= **ìgba**)
- LH 'garden egg' (= **ìgbá**)
- MM 'two hundred' (= **igba**)
- ML 'climbing rope' (= **igbà**)

The current `YORUBA_MINIMAL_PAIRS` says `("igbá", "garden egg", ...)` — **wrong**:
igbá (MH) is 'calabash'; garden egg is ìgbá (LH). Mining caught a factual bug.

Also from the same literature: H is the strongest/most stable tone, M the
weakest (Akinlabi 1985; Pulleyblank 1986; Orie 1997; Akinlabi & Liberman 2000;
Bakare 1995: H has highest F0, highest intensity, shortest duration).
L2 error pattern (Orie 2006b): English-speaking learners use only H and L,
misidentify utterance-initial M as H and utterance-final M as L — directly
usable as diagnostic feedback ("you're flattening mid tones, the classic
English-speaker error").

Nasal contrasts are phonemic (Ajiboye, http://ihafa.unilag.edu.ng/article/download/1259/1006/):
àdá 'cutlass' / àdán 'bat', rù 'carry' / rùn 'smell', ẹrí 'witness' / ẹ̀rín
'laughter', yẹ 'be fit' / yẹn 'that', ìwọ̀ 'hook' / ìwọ̀n 'measuring scale'.

Ekiti dialectology (verified, no invented vocab):
- Èkìtì is a Central Yorùbá dialect (with Ìjẹ̀ṣà, Ìfẹ̀, Mọ̀bà — Awóbùlúyì 1998,
  via Olúmúyìwá, https://core.ac.uk/download/231335179.pdf).
- Aturamu (2024): consonant (esp. /r/) deletion prominent in Standard Yorùbá is
  largely absent in Èkìtì/Òǹdó/Yàgbà — a real, citable dialect feature.
- Arókoyò (2020, https://lasujoh.org.ng/storage/articles/S77CtRP0RlugYEpP2GuNeQUgZtMgN6kwVzPBT1qB.pdf):
  the voiced velar fricative [ɣ] was historically present in Yorùbá and lost by
  oversimplification — background note only, not taught as current Ekiti.
- Dairo (1985, https://ir.oauife.edu.ng/items/fac6d19a-f81c-48ce-be1d-f376757d37af/full):
  Yoruba dialect background changes English pronunciation error patterns —
  justifies the dialect-aware approach.

## 7. Anki interoperability

**Source:** Anki manual, text-file import
(https://github.com/ankitects/anki-manual/blob/HEAD/src/importing/text-files.md):
UTF-8 plain text; `#separator:tab`, `#html:true`, `#notetype:Basic`,
`#deck:<name>`, `#tags:<space-separated>`, `#columns:Front<TAB>Back` headers;
newlines inside fields via `<br>` (real newlines break rows); 3rd column can be
tags (`#tags column:3`).

**Gold mined:** `MistakeNotebook.to_anki_tsv(deck)` and course-quiz export —
one File→Import step and the user's real Anki (incl. AnkiDroid on the phone)
gets the cards.

## 8. Exam simulation — real WAEC/JAMB structures

**Source:** WAEC Physics syllabus scheme
(https://aseiclass.com/catalog/physics_waec.php,
https://myschoolgist.com/32877/PHYSICS.pdf): Paper 1 — 50 multiple-choice,
1¼ hrs, 50 marks; Paper 2 — Section A: 7 short-structured (answer 5, 15 marks)
+ Section B: 5 essays (answer 3, 45 marks), 1½ hrs, 60 marks; Paper 3 —
practical, 3 questions (answer 2), 2¾ hrs, 50 marks. Papers 1+2 taken as one
composite sitting.

**Source:** JAMB UTME format (https://exam-tips.com/exam/jamb-utme,
https://cutoffmark.ng/jamb-utme-2026-score-tips/): 180 questions in 2 hours
(120 min → ~40 s/question); Use of English 60 + 3 subjects × 40; 400 max;
four-option A–D; equal marks; **no negative marking**; CBT only.

**Source:** WAEC Chief Examiners' Reports — 2025 WASSCE weaknesses
(https://citinewsroom.com/2025/12/waec-reveals-major-weaknesses-behind-sharp-dip-in-2025-wassce-results/,
https://www.myjoyonline.com/waec-identifies-seven-key-areas-behind-poor-core-mathematics-performance/):
Core Maths — representing information in diagrams, cumulative frequency
tables/graphs, word problems → mathematical expressions, simple interest,
deductions from real-life situations, Pythagoras, factorization, Spearman's
rank correlation, gradients of straight lines; Science
(https://kuulchat.com/wassce/chief_examiners_report/2019 Science.pdf) —
spelling of technical words (knife, bacteria, manure, wheelbarrow, nutcracker),
greenhouse effect, wave rectification, answering more questions than specified.

**Gold mined:** (a) a `MISCONCEPTION_BANK` keyed by syllabus code, each entry
with `pattern` (what the student says), `correction`, and `source` (the
examiner report) — this is AutoTutor's anticipated-misconception list, grounded
in what Nigerian examiners actually report; (b) an `ExamSim` that runs timed
mocks under the real paper schemes (pacing enforced from the real timings,
no-negative-marking scoring, predicted grade band, pace analysis, misconception
hits); (c) JAMB subject-combination data (Use of English compulsory + 3;
Medicine/Engineering: English+Maths+Physics+Chemistry/Biology etc. from
https://edujects.com/jamb-utme-2026-2027-questions-and-answers/).

## 9. What's deliberately NOT built

- Full FSRS-6 (short-term learning steps, interval fuzz, weight optimization):
  needs per-user fit data; the lite DSR core is the honest subset.
- EM-fitted BKT parameters: need cohort data; ship sensible literature defaults
  (p_init 0.3, p_learn 0.3, p_guess 0.2, p_slip 0.1 — the pyBKT README example
  uses p_T=0.30, p_G=0.10, p_S=0.03, p_L0=0.10) and let them be per-skill
  overridable.
- Per-phoneme tonal alignment: still research-grade (existing honest note kept).
- Invented Ekiti vocabulary: still forbidden; the mi/mo core + cited tone facts
  are the verifiable base.

## 10. Gaps found by inspection (spec floor + inspection)

- `MasteryModel.update()` linear heuristic → BKT posterior (done, §1).
- No forgetting anywhere → HLR half-lives + BKT forget (§1, §3).
- Sessions are in-memory only (restart kills them) → JSON persistence + registry
  save/load.
- Socratic leak: `_fallback_diagnose` tells the student the exact expected
  keywords they're missing ("you're missing: wavelength") — the answer guard
  can't catch it. → sanitize feedback against expected-answer tokens in
  socratic mode.
- `respond()` never records mistakes into the notebook despite the module
  docstring promising it ("Every tutoring mistake is recorded").
- `curriculum.py` classes are not exported from `__init__.py` (dead to the
  chat layer).
- `dialect.py` module-level `from ..voice.money import ...` makes
  `nomorals.learn` unimportable when imported before `nomorals.finance`
  (circular import voice.money ↔ finance.send). → lazy import inside functions;
  sibling files untouched.
- `YORUBA_MINIMAL_PAIRS` igbá error (§6).
- No exam simulation, no study plans, no coverage tracking, no Anki export.
- No due-review interleaving in sessions (retrieval practice is the most
  replicated effect in learning science — Roediger & Butler; FSRS due cards
  feed `next_item()`).
