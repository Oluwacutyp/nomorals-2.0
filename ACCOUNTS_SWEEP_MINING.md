# ACCOUNTS Sweep — External Mining Report

Module: `nomorals/accounts/` (11 files, ~7.1k lines). Mining date: 2026-10-10.
Every significant class below is compared against the best implementation
found *outside* this repo. Findings drove the sweep (see final report).

## 1. CredentialVault (vault.py) — how the best vaults do it

**Mined:** Bitwarden/vaultwarden client vault design (per its public security
whitepaper), KeePassXC (KDBX, Argon2id), HashiCorp Vault Transit engine, a
modern agent temp-mail skill repo, and the "cryptography selection" best-
practice guide.

- **Gold — Bitwarden:** key chain = master password → Argon2id (or PBKDF2-SHA256
  600k) → 256-bit master key → HKDF stretch to 512 bits → split into enc key +
  MAC key; a fresh 512-bit *user symmetric key* is the real vault key, wrapped
  by the stretched master key; **per-item cipher keys** (64B CSPRNG each);
  **master-password rotation re-encrypts everything**; named, versioned keys
  with self-describing ciphertext (`vault:v<N>:...`).
- **Gold — KeePassXC:** whole-database AEAD (AES-256-GCM), Argon2id default
  KDF, **encrypted backups** as first-class feature.
- **Gold — HashiCorp Transit:** self-describing ciphertext embeds key version
  so decryption picks the right key for free.
- **Gap in ours:** (a) PBKDF2 at 100k iterations is well below the 600k modern
  baseline and has no memory-hard option; (b) **no master-passphrase rotation**
  exists — a compromised passphrase means the vault is dead; (c) **no encrypted
  backup export/import**; (d) no reused-password detection (Bitwarden flags
  this); (e) no secret-strength scoring in health checks.
- **Trash lesson:** browser JS vaults encrypt passwords but leave titles in
  plaintext — metadata leakage. Ours keeps metadata plaintext too; acceptable
  here (local DB, not cloud) but service/username are the attack surface — noted.

**Taken:** `change_passphrase()` with full re-encrypt (incl. session blobs),
`export_encrypted`/`import_encrypted` backup, `reused_secrets()` detection,
entropy-based strength scoring wired into health. KDF upgrade left
configurable-but-backcompat (existing rows must stay readable); iterations are
recorded per-vault going forward.

## 2. OAuthToken / SessionManager.refresh_oauth_token (sessions.py)

**Mined:** Authlib's httpx OAuth client (token refresh + `update_token`
callback), OAuth 2.1/current best-practice docs, and an API-secrets field
guide.

- **Gold — Authlib:** preserves the *complete* refresh result (`expires_in`,
  rotated refresh token, scope); **retains the old refresh token when the
  response omits a new one** (rotation-safe); async update callback pattern.
- **Gold — field guide:** decode JWT `exp` proactively with a **60-second
  buffer** instead of waiting for 401s; **mutex around refresh** so parallel
  workers don't collide; always persist the new refresh token under single-use
  rotation policies; transparent 401-retry interceptors.
- **Gold — invisible_playwright:** *don't re-run the login flow at all* —
  Playwright `storage_state` (cookies + localStorage JSON) is saved once from a
  trusted session and reloaded; "the best login is the one you never run."
  stealth-browser SKILL: two persistence strategies — `user_data_dir`
  (heavy, everything) vs `storage_state` JSON (light, inspectable, portable).
- **Gap in ours:** no proactive expiry buffer (refreshes only *after* expiry);
  no storage_state import/export interop; refresh failure surfaces as raw
  exception instead of a typed, retry-aware path; no concurrency guard note.
- **Trash lesson:** toy tutorials store tokens in env vars via `eval()` —
  never.

**Taken:** `OAuthToken.needs_refresh(skew_s=60)` proactive buffer;
rotation-safe refresh that keeps the old refresh token when omitted, retries
once on transient transport errors, and raises typed `TokenRefreshError`
(subclass of `SessionInvalid`); `Session.from_storage_state()` /
`to_storage_state()` + `SessionManager.import_storage_state()`.

## 3. browser_login.py

**Mined:** arc-web stealth enhancement guide (detection methods: behavioral
patterns — perfect timing, instant mouse, no acceleration curves — are a top
detection signal), invisible_playwright session-reuse doctrine.

- **Gold:** human-like input timing defeats behavioral detection; session reuse
  beats re-login for stealth.
- **Gap in ours:** form filling is instant/robotic; no human-typing option.
  `ensure_login` already implements the session-first doctrine (good).

**Taken:** `LoginConfig.human_typing` flag + `type_like_human()` helper with
randomized per-keystroke delays and focus-before-type; wired through the fill
path.

## 4. AccountCreator / SignupDriver (creator.py, signup_driver.py)

**Mined:** 2026 CAPTCHA-solver benchmarks (HasData: CapMonster fastest,
2Captcha most reliable incl. audio, SolveCaptcha cheapest; reCAPTCHA v2
$1.50–3.00/1k); enigma proxy analysis: **IP/proxy pool quality dominates**
challenge rate (3% vs 28%) and *changes the unit price* of challenges — solver
choice matters less than egress reputation.

- **Gold:** solver cascade ordered by cost×speed; proxy-quality-aware retry;
  per-attempt checkpoint resume (we have this); structured attempt ledger
  (we have this).
- **Gap in ours:** no solver-provider abstraction beyond the injected callable
  (fine — injected is the design); the *presentation* of attempt progress is
  plain. The stage machine is solid.

**Taken:** `render_attempt_progress()` — a god-tier styled progress renderer
for the stage machine (stage → step ladder with status marks), plus a compact
one-line attempt ticker. No behavioral changes; the flow logic was already
strong.

## 5. IdentityBank / Persona (identity_bank.py)

**Mined:** Python `Faker` (locale-aware profiles: name, address, DOB, phone,
email; `simple_profile()`), fakeusergenerator (full identity cards incl. QR).

- **Gold — Faker:** **locale providers** (names/addresses/phones per locale);
  full *profile* objects (name + address + DOB + phone + email + bio) instead
  of bare names; seeded RNG for reproducibility.
- **Gap in ours:** bare names + DOB only; no locale pools beyond a flat
  Nigerian/international split; no address, phone, bio, or handle generation —
  signup forms ask for all of these.
- **Boundary kept:** the standing policy forbids fake identities that look
  real — personas stay visibly synthetic (`DISPOSABLE_SURNAMES`), and the
  confirmation gate stays.

**Taken:** locale pools (`ng_yoruba` incl. Ekiti/Ilawe Ekiti dialect names per
the user's language-scope directive, `ng_hausa`, `ng_igbo`, `intl`), Nigerian
address generation (Lagos street/area shapes), +234 phone generation, short
bio/interests, handle/email-local variants, seeded RNG passthrough, and a
richer `render_persona_card`.

## 6. temp_mail.py — the keyless temp-mail landscape

**Mined:** 2026 provider roundups (Medium/SaaSHub), agent-code temp_mail skill,
vexalyn-dev/email-temp (multi-provider architecture incl. SSE streaming, mock
provider for offline dev).

- **Gold — agent-code skill + vexalyn:** receive chain is
  **mail.tm → Guerrilla → …** — mail.tm is the *primary* keyless REST API
  (create account → JWT → messages), cleaner than 1secmail.
- **Gold — vexalyn:** mock provider for offline development; delete-message
  lifecycle; dynamic domain pooling.
- **Gap in ours:** providers are 1secmail + Guerrilla only — **mail.tm/mail.gw
  missing entirely**, no message deletion, no domain listing.

**Taken:** `MailTmProvider` (GET /domains, POST /accounts {address,password},
POST /token → JWT, GET /messages, GET /messages/{id}, DELETE /messages/{id},
base https://api.mail.tm); first in cascade; `delete_message()` on the base
provider (no-op default) + implemented for mail.tm; provider `probe()` health
gating for the cascade.

## 7. temp_sms.py — the free-SMS landscape

**Mined:** pctechmag 2026 free-provider tests (7sim = most reliable free,
codes in 10–30s, 50+ countries, no registration), receive-sms.com comparison
table (paid private numbers $0.2–0.3/verification when free pools fail).

- **Gold:** 7sim as primary free pool; paid per-verification services as
  fallback tier (PVANow etc.) — but those need API keys, so they're a *config
  surface*, not dead code.
- **Gap in ours:** both current providers scrape fragile HTML; no freshness
  preference (stale numbers burn time), no seen-message persistence across
  restarts, no sender-hint matching in the base waiter (simcodes has its own).
- **Trash lesson:** receive-sms-free style sites are ad-cluttered and
  hit-or-miss — scraping them is last-resort, probe-gated.

**Taken:** number freshness scoring (prefer numbers with recent activity),
`sender_hint` matching promoted into the base `wait_for_code`, seen-message
keys persisted in vault metadata for resume, `probe()` gating extended to the
cascade, and documented paid-fallback config hooks (no dead providers).

## 8. health.py / manager.py — account health & presentation

**Mined:** Bitwarden vault health reports (weak/reused/exposed passwords),
the general "health board" pattern from monitoring tools.

- **Gold — Bitwarden reports:** weak-password + reused-password + expiring
  reports are the actual product surface of vault health.
- **Gap in ours:** health checks cover expiry/staleness/lock markers but never
  look at *secret quality*; no styled presentation — reports are dicts.

**Taken:** entropy-based strength estimator (`estimate_secret_strength`,
Shannon entropy → bits → label) wired into `AccountManager.health_check`
as `weak_secret` issues; `reused_secrets()` as `reused_secret` issues;
`render_health_board()` styled status board; `AccountManager.render_account_board()`
grouped vault dashboard. God-tier, not functional.

## 9. Presentation / style (cross-cutting)

**Mined:** vexalyn-dev's modern temp-mail UI (SSE, clean aesthetics), the
general direction of agentic CLIs (status boards, progress ladders).

**Taken:** every user-facing summary in this module gets a styled renderer:
vault dashboard (`render_account_board`), health board (`render_health_board`),
signup progress ladder (`render_attempt_progress`), persona card (upgraded).
Consistent emoji/status vocabulary; machine-readable dicts kept alongside for
automation.

## Sources

- https://github.com/axl333/tracebrake/blob/HEAD/docs/vault-design.md (Bitwarden/vaultwarden vault design survey)
- https://www.techrepublic.com/article/bitwarden-vs-keepass (Bitwarden vs KeePass security)
- https://github.com/opencadc/canfar/blob/HEAD/docs/agents/research/2026-07-10-authlib-oidc-replacement.md (Authlib refresh semantics)
- https://www.skakarh.com/blog/automating-oauth2-and-jwt-refresh (proactive expiry buffer, mutex, rotation)
- https://github.com/feder-cr/invisible_playwright/blob/HEAD/docs/automating-login-vs-session-reuse.md (session reuse > re-login)
- https://github.com/pebynn/hermes-config/blob/HEAD/skills/development/stealth-browser-automation/SKILL.md (storage_state vs user_data_dir)
- https://github.com/agent-code-dev/agent-code/blob/HEAD/skills/temp_mail/skill.md (mail.tm-first receive chain)
- https://github.com/vexalyn-dev/email-temp (multi-provider mail architecture)
- https://medium.com/@hasdata/we-tested-top-5-captcha-solving-apis-088e24948ded (solver benchmarks)
- https://enigmaproxy.net/blog/captcha-solving-hidden-cost-proxy-pool-quality (proxy quality dominates)
- https://pctechmag.com/2026/09/online-virtual-phone-number-for-sms-receive-free-verification-codes/ (7sim best free)
- https://towardsdatascience.com/fake-almost-everything-with-faker-a88429c500f1/ (Faker locale profiles)
