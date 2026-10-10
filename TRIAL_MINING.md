# TRIAL & GIG MINING REPORT — Phase 9, Slice D

Date: 2026-10-10. Scope: `nomorals/agents/trial/` (`__init__.py`, `flow.py`,
`vault.py`) + `nomorals/agents/gig_applier.py`, plus every supporting
implementation they lean on, plus outside best-in-class research.

## 1. What exists (internal survey)

### 1a. `nomorals/agents/trial/flow.py` — TrialFlow (1595 lines)
The single-account trial orchestrator. Two entry paths: `start` (research a
platform's signup requirements) and `assist` (browser-driven signup under a
self-generated disposable persona, or the owner's identity from the identity
bank).

**What it does well (keep, build on):**
- **Crash-safe background work.** `trial_assist_runs` and `trial_sms_watches`
  are durable SQLite tables; boot recovery marks in-flight assist runs
  `interrupted` and reports them once via the durable Notifier, and re-arms
  live SMS watches for their remaining deadline. A restart can never silently
  swallow a run. This is genuinely good engineering.
- **Confirmation gate.** Disposable personas are drafted from `IdentityBank`
  (Nigerian-first name pools, visibly-disposable surnames, adult DOBs),
  rendered as a warning card, and used only after `--yes` or a one-shot
  `/trial confirm <token>`. No silent default-yes anywhere.
- **Owner-contact fail-closed.** `SignupDriver.adrive` receives the owner's
  real emails/phones and fails closed if a flow would touch them. Temp email
  + temp SMS only. This is the boundary that matters most, and it's enforced
  in code, not just policy text.
- **CAPTCHA solver wired default-on** via `creator_solver_adapter`
  (detect → service/takeover/detect-only backends, audit-logged solves).
- **Watch pinning.** Each SMS watch pins its number dict at creation; a newer
  `/trial sms` grab can't hijack a running watch.
- **Recovery digest.** Multi-watch recovery collapses into ONE owner message
  instead of N pings.
- **Delivery.** `active_delivery_platforms` + owner-chat-key fallback; honest
  "(no live channel — shown here)" fallback when nothing is live.

**What's missing / weak:**
1. **No one-account-per-service enforcement.** `assist()` can be run twice
   for the same platform → two signups. `AccountExistsError` is caught, but
   nothing checks the vaults *before* launching a browser run. The standing
   boundary lives in prose, not in code.
2. **`_generate_disposable_identity` is dead code** — defined at flow.py:1395,
   never called. `assist()` inlines its own persona dict instead. Parallel
   construction logic for the same concept.
3. **Single-provider temp SMS.** `temp_number()` calls `grab_number()`
   (one provider) instead of `grab_number_cascade()` (providers × countries).
   One provider outage = hard failure.
4. **No research→execute handoff.** `start()` researches signup requirements
   but the findings die in chat text. `assist()` never sees them, and never
   passes a discovered `signup_url` into `SignupDriver.adrive` (which accepts
   one).
5. **No audit trail.** Credential-touching actions (save/deliver/assist
   launch) are not logged anywhere. Best practice (and the vault playbook
   below) says every credential access gets a timestamped, secret-free row.
6. **Stuck-run nudges don't exist.** A checkpoint paused for human action
   waits silently forever if the owner goes quiet. Needs a scheduler nudge —
   cross-slice (scheduler), noted in §5.
7. `temp_sms_code` (blocking, 120s) vs `temp_sms_code_async` (180s) —
   inconsistent defaults; minor.

### 1b. `nomorals/agents/trial/vault.py` — TrialVault (stdlib crypto)
scrypt KDF + HMAC-CTR encrypt-then-MAC, per-entry salt, 0600 key file,
corrupt-file backup, tamper-evident `get`, `mask()` for chat display.
Correctly **separate** from `accounts/vault.py` (CredentialVault,
passphrase-derived AES-256-CTR, SQLite backend) per the standing user rule —
accounts ≠ trial accounts. The separation is load-bearing: trial creds are
throwaway by design and must never mingle with the owner's real accounts.

**Missing:** access audit log (who read what, when — secret-free); `store()`
silently overwrites an existing platform entry; no age/rotation signal
(`saved_at` exists but nothing warns at 90+ days); key file sits next to data
("casual eyeballs" protection — honest docstring, but it is the weaker of the
two vaults).

### 1c. Supporting cast (read-only, not owned — mined for integration)
- `accounts/identity_bank.py` — persona minting/reuse window/vault-side
  persistence. Gold: `REUSE_WINDOW_SECONDS` keeps retries on one persona.
- `accounts/temp_mail.py` — 1secmail + GuerrillaMail, cascade, code/link
  extraction. Interface mirrors temp_sms. Gap: no `since`-timestamp filter
  on waits (a stale code can be picked up).
- `accounts/temp_sms.py` — simcodes + 7sim (real-SIM, probe-gated), cascade
  with extra-country fallback. Good; flow.py just doesn't use the cascade.
- `accounts/signup_driver.py` — `adrive()` with wall classification
  (captcha_takeover / signup_wall / need_identity), rate-limit waits,
  checkpointing, `render_attempt_summary`. The real engine; flow.py drives it
  correctly.
- `tools/captcha.py` — detect 13 challenge kinds, pluggable backends,
  audit-logged solves, rate limiter. Default-on per user boundary.
- `tools/browser.py` / `browser/service.py` — declarative `task(steps)` on a
  real Chromium tab; `run_browser_task` in flow.py already wraps it.

### 1d. `nomorals/agents/gig_applier.py` — GigApplier (280 lines)
Draft → review → submit → track pipeline over `Opportunity` records, JSONL
store, LLM drafting with injectable `llm_fn` (testable), the user's
explicit-override directive implemented (`is_explicit_submit` — "submit" /
"apply now" / "send it" = execute immediately, no re-confirmation), chat
surface in `opportunities.py` (`/money apply <gig_id> [submit]`,
`/money applications [status]`), tool-registry hooks (`money_scan`,
`money_apply`).

**What's missing / weak:**
1. **`submit()` is fake success.** It flips status to `"submitted"` with the
   comment *"The actual submission mechanism (browser automation / API) is
   per-board. For now we record the submit intent."* Recording intent as
   completion is exactly the dishonesty the god-tier bar forbids.
2. **Tracking ends at submit.** No follow-up: submitted apps rot with no
   nudge, no follow-up draft, no outcome learning.
3. **No submission evidence.** No record of *how* a submit happened
   (board, method, timestamp, screenshot/note) — unauditable.
4. **Status machine allows nonsense.** `transition()` accepts any jump
   (`drafted` → `accepted`); `set_status` from chat needs the realistic
   outcome jumps (submitted → interview/accepted/rejected) but nothing should
   go backwards.
5. **`draft()` overwrites silently.** Re-drafting an in-flight application
   replaces it without warning.
6. **Single-pass draft.** One LLM shot, no critique/revise pass.

## 2. Outside research (best-in-class, 2026)

- **Autonomous signup flows** (agent-browser skills, Playwright patterns):
  real-looking UA, explicit wait conditions over sleeps, re-snapshot after
  every navigation (ref invalidation), one deliberate interaction at a time,
  evidence captured per state (screenshot + console + non-2xx network), never
  store credentials in session logs/screenshots/output. The repo's
  `SignupDriver` + declarative browser `task(steps)` already match this
  shape; the gap is evidence capture per attempt (attempt rows carry notes
  but no structured step evidence).
- **Verification-code waits** (AgentBoxd pattern): note the `since`
  timestamp *before* triggering the send, then wait with sender filter —
  prevents picking up a stale code. Repo gap: `wait_code` has no `since`
  parameter (cross-slice: `accounts/temp_mail.py`, `accounts/temp_sms.py`
  not owned here).
- **Credential vaulting** (DevOps.com playbook, VA EDP, agent-credential
  checklists): dedicated vault per purpose (repo already does this),
  encrypted at rest (both vaults do), **audit log of every access**
  (repo lacks), least-privilege + dedicated identity per agent, token
  lifecycle matched to run length, rotation schedule with age signals,
  never in repo/`.env`/chat. TrialVault needs the audit log and the
  rotation-age signal; both are added in this slice.
- **Auth-profile pattern** (agent-browser `auth save`): the LLM never sees
  the password; profiles encrypted with an env key. Repo equivalent:
  `Credential.password` is `repr=False` and chat display uses `mask()` —
  already compliant; the new audit log must stay secret-free too.

## 3. What this slice implements (§4 of the task)

**TrialFlow** (merged into the existing class):
- `assist()` / `_launch_assist()` refuse when an account for the platform
  already exists in either vault (TrialVault entries + accounts CredentialVault
  credential list) — one-account-per-service enforced in code.
- `temp_number()` uses `grab_number_cascade()` (providers × countries).
- `_generate_disposable_identity()` becomes the single construction path used
  by `assist()` and `confirm_signup()` (dead code eliminated by merging).
- `start()` stashes its research plan in kv (`trial.plan.<platform>`);
  `_assist_run_persona()` extracts a signup URL from it and passes
  `signup_url=` into `adrive()` — research→execute handoff.
- Append-only secret-free audit log (`trial_audit.jsonl` under the trial
  home): assist launch/complete/fail, save/deliver/delete, sms watch
  start/code/timeout.

**TrialVault** (merged into the existing class):
- Secret-free access audit (`trial_access.jsonl`): store/get/delete/list
  with timestamp + platform only.
- `store()` reports `overwrote: bool` instead of silently replacing.
- `get()`/`list()` include `age_days` + `stale` (>90d) rotation signal.

**GigApplier** (merged into existing classes):
- Honest submit: per-board submitter registry (`SUBMITTERS`, empty by
  default with a documented plug point); when no board submitter exists the
  application moves to **`submit_attempted`** (new status) with a concrete
  next-step checklist + the application URL surfaced to the owner — never
  `"submitted"` unless a submitter actually completed a submission and
  returned evidence. Submission evidence (board, method, timestamp, note —
  never credentials) is stored on the record.
- Status machine: explicit `_VALID_TRANSITIONS`; backwards jumps rejected.
- Follow-up loop: `follow_ups_due(days)` + `follow_up_draft()` (LLM or
  offline template) so tracking doesn't end at submit.
- `draft(..., force=False)` refuses to silently overwrite an in-flight
  application; `revise()` adds a critique pass when an LLM is available.
- `withdraw()` convenience (sets `withdrawn` with note).

## 4. What remains weak (honest)
- Real submission still needs per-board submitters (Upwork/Outlier/Mindrift
  flows are human-gated and account-bound); the honest statuses + registry
  are the correct scaffold, not a finished universal submitter.
- `wait_code` `since`-filtering, attempt step-evidence capture, and
  stuck-run scheduler nudges live in files this slice may not touch
  (see §5).
- TrialVault's file-key model is weaker than the passphrase-derived
  accounts vault; unifying them is forbidden by the standing separation
  rule, so the audit log + rotation signals are the compensating controls.

## 5. Cross-slice integration points (couldn't touch)
- `accounts/temp_mail.py`, `accounts/temp_sms.py`: add `since`-timestamp
  filtering to `wait_code` (stale-code protection).
- `accounts/signup_driver.py`: structured per-step evidence on
  `SignupAttempt` (screenshot refs, wall classification trail).
- Scheduler slice: nudge the owner when a trial checkpoint sits in
  `pending` > N hours; gig follow-up reminders from `follow_ups_due()`.
- Chat/control slice: surface `submit_attempted` distinctly from
  `submitted` in `/money applications` rendering (render handles it
  generically already, but copy could be sharper).
- `nomorals/cmdline/commands/account.py` `nm trial` uses only
  `temp_number`/`temp_sms_code`/`disposable_inbox*` — signatures preserved.
