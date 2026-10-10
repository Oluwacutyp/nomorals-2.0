# CONNECTORS SWEEP — External Mining Report

Module: `nomorals/connectors/` (45 files). Mined 2026-10-10. This report was
written BEFORE any code changes. Every significant class was compared against
the best implementation found outside the repo — best AND trash.

## Method

For each class cluster: "How does the best implementation of X do it?" then
"What features SHOULD this have that it doesn't?" and "How should this
LOOK/FEEL?" Sources are summarized in my own words; no source text is
reproduced beyond brief fair-use quotes.

---

## 1. The framework spine: `base.py` / `registry.py` / `auth.py`

### Best: Home Assistant integration framework
HA is the best-in-class *service connector framework* in the open-source
world (thousands of integrations). Its patterns:

- **Config entry lifecycle**: setup → config flow → reauth flow → options
  reload. Duplicate avoidance via `unique_id` set early. Reauth asks *only*
  for the changed credential and updates the existing entry in place.
- **DataUpdateCoordinator**: one poller per integration; entities subscribe.
  Never poll per-entity. Ours has no shared polling/caching primitive at all.
- **Error taxonomy with semantics**: `ConfigEntryAuthFailed` (triggers the
  reauth flow automatically), `ConfigEntryNotReady` (retry later),
  `UpdateFailed(retry_after=60)` (rate-limit backoff). Errors *drive
  behavior*, they aren't just messages.
- **Device info with stable identifiers**; entities are created only for
  data that actually exists; entities report `unavailable` when offline.
- Quality tiers (Bronze→Gold) make "done" checkable — config flow tested,
  proper error handling, translations.

**Gap in ours**: `ConnectorError` is flat — one class for auth failures,
rate limits, network blips, and validation errors. Callers can't
distinguish "token dead, re-auth" from "rate limited, back off" from
"server down, retry". ccxt (see below) proves a rich error hierarchy is
the thing that makes a connector library *operational* rather than
decorative.

### Best: ccxt (108+ exchanges, 40k+ stars) — the unified-API gold standard
- **Unified interface**: `fetchTicker`, `createOrder`, `fetchBalance` work
  *identically* across every exchange. Success = consistent method names.
- **Error hierarchy**: `NetworkError`, `ExchangeError`,
  `AuthenticationError`, `InvalidOrder`, `RateLimitExceeded` — each maps to
  a retry/re-auth/abort decision.
- **Rate limiting**: leaky bucket (default) + rolling window, configurable
  per exchange/method/request (configuration cascading).
- **Escape hatches at every level**: unified API → `request()` with auto
  signing → `raw_request()` without signing. Users are never trapped.
- "No silent fallbacks — fail loudly with actionable errors."

**Gap in ours**: no shared retry/backoff primitive (each connector
hand-rolls `Retry-After` parsing — discord.py and slack.py both have
private `_retry_after`), no pagination helper (every list method
hand-rolls page loops), no unified error taxonomy.

### Best: n8n node design
- Resource + operation pattern; `displayName`/`icon`/`description` manifest;
  `usableAsTool` flag so nodes double as AI-agent tools; a **credentials
  test function** returning `{success, message}`; subtitle showing the
  current operation.
- Node list supports a `query` filter.

**Gap in ours**: `registry.list_connectors()` has no search; no
`describe` manifest merging capabilities; no fleet-wide health snapshot
(HA's device-health view / n8n's credential test, but across *all*
connectors at once). The `nm connectors list` output is a bare dump.

### Trash mined
Tutorial-grade "connector frameworks" that are one `requests.get` wrapped
in a class with `connect()` that just stores a string. Also: frameworks
that log secrets on error paths (ours already scrubs — keep that bar).

### → Framework decisions
1. **Error taxonomy** in `base.py`: `ConnectorAuthError`,
   `ConnectorRateLimitError` (carries `retry_after`), `ConnectorNetworkError`,
   `ConnectorNotFoundError`, `ConnectorValidationError` — all subclasses of
   `ConnectorError` (backward compatible; existing `XError` classes keep
   working, new code should raise these).
2. **Shared `request_with_retry()`** in `base.py`: exponential backoff +
   jitter, honors `Retry-After`, retries 429/502/503/504, fail-loud on
   4xx. Opt-in per connector.
3. **Shared `paginate()`** helper: Link-header (`rel="next"`) and
   page-number styles in one generator.
4. **Registry upgrades**: `search_connectors(query)`,
   `describe_connector(id)` (full manifest: auth, provisionable,
   capabilities, category), `health_snapshot()` (fleet-wide status with
   per-connector timeouts — the dashboard primitive), category tags.
5. **`capabilities()` contract** on `Connector`: several connectors already
   define ad-hoc `capabilities()` dicts (duffel, jumia, leonardo,
   nanobanana, googleflow) with different shapes. Standardize the base
   contract; keep theirs working.

---

## 2. Auth: `auth.py` / `_google_oauth.py` / `checkpoints.py`

### Best: OAuth 2.0 PKCE practice (RFC 8252 §8, RFC 9700 baseline)
- System browser only (never an embedded webview); `127.0.0.1` literal,
  never `localhost`.
- **PKCE S256 mandatory for public clients, recommended for confidential
  ones**; verifier 43–128 chars, transaction-specific.
- `state` validated *before* anything else in the callback is accepted.
- Authorization codes short-lived, single-use; second redemption is a
  compromise signal.
- Refresh: scheduled at expiry−5 min, single-flight, keep the old refresh
  token when the response omits one; revoke best-effort on disconnect,
  then wipe.
- Tokens never in logs or error objects (ours already does this — hold).

**Gap in ours**: `GoogleOAuth.google_loopback_flow` has **no PKCE** and **no
state validation** — the two cheapest hardening wins in the OAuth world.
`device_flow_token` is solid (RFC 8628, handles `slow_down`).

### → Auth decisions
1. `auth.py`: add `pkce_pair()` (S256) and `new_state()` helpers.
2. `_google_oauth.py`: `google_authorize_url(..., code_challenge=..., state=...)`
   and `google_loopback_flow(..., use_pkce=True)` — generate verifier,
   send challenge, validate state on callback, send verifier at exchange.
   Backward compatible (opt-out flag).
3. `checkpoints.py`: add `history()` (non-pending checkpoints) — the audit
   trail the dashboard needs. The persist→ping→wait/raise design is already
   best-in-class; keep.

---

## 3. Webhooks: Paystack/Stripe/GitHub

### Best: Stripe webhook + idempotency practice (2026 consensus)
- **Idempotency-Key on every mutating call**, and the key must be *stable
  per logical operation* — derived from business IDs, not a fresh UUID per
  request (a per-request UUID defeats the purpose on retry).
- Webhook signature verification is mandatory: HMAC-SHA256 over
  `timestamp.payload` with the *raw* body (never re-serialized JSON);
  reject timestamps older than ~5 min (replay mitigation); constant-time
  compare.
- At-least-once delivery → **dedupe by `event.id` in the same transaction**
  as the business work; 5xx triggers retries, 4xx doesn't; return 2xx fast,
  process async.
- Pin the API version; never log full objects (PII); restricted keys.

**Gap in ours**: `stripe.py` sends **no Idempotency-Key** on any mutation
(duplicate charge on network retry is the exact failure Stripe's docs
warn about). Paystack has `verify_webhook_signature`; Stripe has *none*;
GitHub can *create* webhooks but cannot *verify* inbound ones. Three
different HMAC schemes hand-rolled in three places.

### → Webhook decisions
New `webhooks.py`: one `verify_signature(provider, payload, signature,
secret)` for `stripe` / `paystack` / `github` (timestamped Stripe scheme
with replay window; constant-time compare everywhere), `parse_event()` →
normalized `WebhookEvent(id, type, created_at, data)`, and a TTL
`WebhookDeduper` for at-least-once dedupe. Existing
`PaystackConnector.verify_webhook_signature` delegates to it (no parallel
system); Stripe and GitHub gain verify methods on the same primitive.

---

## 4. Proxy pool: `proxypool.py` (41 KB, the deepest infra here)

### Best: proxyhive (Go) / proxyhub / production proxy practice
- **Rotation strategies**: P2C — power of two choices — (default),
  round-robin, random, least-latency. P2C avoids hotspotting better than
  naive round-robin.
- **Health scoring**: `score = success_rate * 0.6 + latency_score * 0.4`
  (proxyhub); **EWMA latency tracking** (proxyhive) instead of raw last
  sample; latency→score curve like
  `max(0, min(100, 100*(2000-latency_ms)/1900))`.
- **Sticky sessions**: pin a key → proxy for a TTL (login flows, carts).
- **Identity discipline** (production writeups): rotate the *whole identity*
  (IP + cookies + fingerprint), never the IP alone mid-session — a coherent
  session made incoherent is *more* suspicious.
- **Ban detection**: classify responses DATA | BLOCK | CHALLENGE | THROTTLE;
  track bans per subnet/ASN (bans cluster); quarantine, don't just drop.
- Automatic exponential backoff + recovery; geo-filtering by
  country/city/ASN; pool stats (total/alive/dead/avg latency).

**Gap in ours**: `rotate()` exists but is strategy-less (single policy);
no sticky sessions; health is binary-ish (no EWMA, no 0–100 score);
no geo filter on selection; no quarantine-with-cooldown primitive; no
`stats()` rollup.

### → Proxy decisions
Extend the real class: `select(strategy=...)` with
`round_robin | random | least_latency | p2c`, `sticky_acquire(key, ttl)` /
`sticky_release(key)`, EWMA latency + 0–100 health score in
`_record_health`, `quarantine(proxy_id, seconds)`, `stats()` rollup, country
tag on `add_proxy` with geo filtering on select. No parallel pool.

---

## 5. GitHub: `github.py` — the coding agent's connector

### Best: PyGithub / Octokit patterns
- Issues and PRs share the issues endpoint; PRs identifiable by the
  `pull_request` key — **filter them out when listing issues**.
- `per_page=100` + `octokit.paginate` for large repos; Link-header walk.
- 403 → check `X-RateLimit-Remaining`, conditional requests with ETags;
  422 → read the `errors` array; secondary rate limits → 1s delay between
  mutations.
- File contents API: get (base64 + sha) → create/update with sha for
  optimistic concurrency.
- PR flow: create (head/base/title/body) → list files → merge (method:
  merge/squash/rebase).

**Gap in ours**: `github.py` has repos/branches/releases/webhooks/deploy
keys/push/backup — but **zero issue/PR/file-content operations**. For
Devon-the-coding-agent this is the single biggest capability hole in the
whole module. `gh` CLI does all of this; our connector should too.

### → GitHub decisions
ADD: `list_issues` (PR-filtered), `get_issue`, `create_issue`,
`comment_issue`, `list_pull_requests`, `get_pull_request`,
`create_pull_request`, `merge_pull_request`, `list_pull_files`,
`get_file_content`, `create_or_update_file`, `search_repositories`,
`search_issues`. Built on the shared `paginate()` helper. No new client —
extend the class.

---

## 6. Telegram Bot API: `telegram.py`

### Best: PTB / aiogram / grammY skill consensus
Core method surface every serious wrapper exposes: `sendMessage`,
`sendPhoto` / `sendVideo` / `sendDocument` / `sendMediaGroup`,
`editMessageText`, `deleteMessage`, `answerCallbackQuery`, `getChat`,
`setMyCommands`, `setWebhook`/`deleteWebhook`/`getWebhookInfo`.
Rate limits: ~1 msg/sec per chat, ~30/sec global, 20/min per group;
**429 returns `retry_after` — honor it, don't hammer**.
Inline keyboards: `callback_data` ≤ 64 bytes. Webhook endpoints validate
`X-Telegram-Bot-Api-Secret-Token`.

**Gap in ours**: only `send_message`, `get_updates`, `get_chat`,
`leave_chat`. No media sends at all (a bot that can't send a photo is a
toy), no edits/deletes, no callback-query answering (so no inline
keyboards), no webhook management. 429 currently raises instead of
surfacing `retry_after`.

### → Telegram decisions
ADD: `send_photo`, `send_document`, `send_video`, `send_media_group`
(multipart via `HttpClient.post_multipart`), `edit_message_text`,
`delete_message`, `answer_callback_query`, `set_webhook`,
`delete_webhook`, `get_webhook_info`. Surface `retry_after` on the 429
path (raise `ConnectorRateLimitError` with `retry_after` set — the new
taxonomy paying off immediately).

---

## 7. Trading: `binance.py` / `coinbase.py` / `exness.py`

### Best: ccxt applied to trading connectors
Unified names (`get_price`≈`fetchTicker`, `place_order`≈`createOrder`,
`get_balances`≈`fetchBalance` — ours already rhymes with this, good),
sandbox mode for testing, `adjustForTimeDifference` clock handling
(ours: exness already learns clock offset — keep), error hierarchy
separating auth vs invalid-order vs network.

**Gap**: ours are already deep (42 KB exness with ed25519 hand-rolled,
checkpoint-gated orders). The real gap is cross-cutting: no shared
**order preview / risk gate** language and no unified market-data shape
across the three. Judgment call: unifying shapes across three live trading
connectors is a breaking change to agent-facing contracts — out of scope
for this sweep. The taxonomy + retry + idempotency work covers the
operational gaps. No changes here beyond adopting new error classes where
they already raise equivalents.

---

## 8. Presentation: how connectors LOOK/FEEL

### Best: `rich`/`textual` CLIs, n8n credential test UX, HA device pages
- One theme: semantic styles (success/error/warning/info), icons in one
  place, TTY detection, NO_COLOR support. (Our `nomorals/cmdline/style.py`
  already implements exactly this — the presentation layer should *use*
  it, not reinvent it.)
- Status displays: colored state dot + account + last-checked + one-line
  detail; fleet views roll up to counts (HA: "3 unavailable").
- Connect flows: numbered steps, what-you-need-up-front, what gets stored
  where.

**Gap in ours**: connector output is bare `print`/dict dumps. `nm
connectors status` prints one flat line. There is no styled fleet view,
no capability sheet, no connect guide renderer anywhere in the package.

### → Presentation decisions
New `present.py` in the connectors package: `render_connector_table()`,
`render_status()`, `render_health_snapshot()`, `render_capabilities()`,
`render_connect_guide()` — all returning strings, all built on
`nomorals.cmdline.style` (TTY-aware, ASCII fallback, NO_COLOR-safe), all
pure functions (testable, CLI-adoptable later without touching contracts).

---

## 9. What I am NOT changing (mined, judged out of scope)

- **Trading connectors' public shapes** (binance/coinbase/exness): deep and
  live; unifying their market-data shapes would break agent contracts.
- **Jumia/Konga/Jiji scrapers**: mined; they're necessarily site-specific.
  No generic gold to merge beyond retry/backoff, which they get via base.
- **Spotify/SoundCloud/YouTube/AudD**: feature-complete for the playback
  wiring; untouched.
- **Mono/Plaid/Wise/Paystack money-movement**: audited surface is already
  deep; changes limited to webhook/idempotency primitives, no flow
  redesigns (money code churns only with a reason).

## Implementation checklist (maps to code)

- [ ] `base.py`: error taxonomy (5 subclasses), `request_with_retry()`,
  `paginate()`, `capabilities()` contract, `CATEGORY` attr
- [ ] `registry.py`: `search_connectors`, `describe_connector`,
  `health_snapshot`, `connector_categories`
- [ ] `auth.py`: `pkce_pair()`, `new_state()`
- [ ] `_google_oauth.py`: PKCE + state in authorize URL + loopback flow
- [ ] NEW `webhooks.py`: verify/parse/dedupe; wire paystack/stripe/github
- [ ] NEW `present.py`: styled fleet/status/capability/guide renderers
- [ ] `github.py`: issues, PRs, file contents, search
- [ ] `telegram.py`: media sends, edits, callbacks, webhooks, 429 taxonomy
- [ ] `proxypool.py`: strategies, sticky sessions, EWMA score, quarantine, stats, geo
- [ ] `stripe.py`: Idempotency-Key on mutations, webhook verify/parse
- [ ] `checkpoints.py`: `history()`
- [ ] `__init__.py`: export new surface
- [ ] `tests/test_connectors_sweep.py`: real tests for all of the above
- [ ] commit `sweep(connectors): …`, push `two main:main`
