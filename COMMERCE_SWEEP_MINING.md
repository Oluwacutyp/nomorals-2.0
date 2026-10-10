# COMMERCE Sweep — External Mining Report

Module: `nomorals/commerce/` (3 files: `__init__`, `cart_recovery`, `medusa`).
Date: 2026-10-10. Written BEFORE any implementation, per sweep method.

Every significant class was compared against the best implementations outside
this repo. "Best" = mined for gold to merge. "Trash" = mined to learn what NOT
to do. Each section ends with the gaps this sweep will fill.

---

## 1. `CartRecovery` — abandoned-cart → WhatsApp sequence

### How the best do it

**Timing (Baymard Institute, Klaviyo, practitioner consensus):**
- ~70.22% of carts are abandoned (Baymard meta-analysis of 50 studies).
- The canonical 3-message sequence: **~1h (plain reminder, NO discount)** →
  **~24h (objection handling: shipping cost, returns, trust)** →
  **~48-72h (urgency + incentive as last resort)**. Sources:
  https://semarkglobal.com/blog/abandoned-cart-post-purchase-email-flows-recover-revenue
  https://www.hustlermarketing.com/how-to-set-up-klaviyo-flows-the-complete-automation-guide-2026/
  http://topgrowthmarketing.com/how-can-i-optimize-my-abandoned-cart-emails-in-klaviyo/
- The #1 cited abandonment reason is **unexpected extra costs at checkout**
  (39-48% — Baymard); message 2 must address it, not just repeat the nudge.
- Klaviyo benchmarks: abandoned-cart flows earn the **highest revenue per
  recipient of any flow type ($3.65 avg, $28.89 top-10%)** with ~50.5% open
  rates and ~5-15% recovery rates (dtcskills Klaviyo skill bible).
- SMS/WhatsApp-class channels: 98% open rate, read within ~90 seconds, convert
  15-20% — our WhatsApp sender is the right channel, and 30-min first touch is
  defensible for messaging (faster reads than email); step 2/3 cadence should
  follow the 24h/48-72h consensus.
  https://www.ringly.io/blog/ecommerce-cart-abandonment-statistics-2026

**Discount discipline (Klaviyo best practice):**
- NEVER lead with a discount — it trains customers to abandon intentionally.
  Reserve the incentive for the final message, and only where margin allows
  (high-value carts). 40-60% of recoveries happen before a discount is needed.
- **Our gap:** step 3 has no incentive mechanism at all. Gold: an optional
  coupon reserved at intake and rendered ONLY in the last call.

**WhatsApp Business compliance (Meta policy — real, enforced):**
- Cart-recovery messages are **marketing** messages (per Meta's 4 categories:
  marketing / utility / authentication / service). Marketing = template-only,
  **explicit opt-in required** (documented consent naming the business;
  pre-checked boxes and bulk lists do NOT qualify), charged per message.
  https://www.auditsocials.com/blog/whatsapp-business-api-click-to-whatsapp-ads-compliance-guide-2026
  https://dev.to/dawnofgenx/the-complete-guide-to-whatsapp-business-api-pricing-in-india-2026-1hbh
- Outside the 24-hour customer-service window only pre-approved template
  messages may be sent.
  https://github.com/waseemnasir2k26/n8n-whatsapp-compliant-agent
- **Opt-out lines reduce bans:** a recipient who can reply STOP has less
  reason to tap "Report spam"; quality rating (green/yellow/red) governs
  sending limits and bans.
  https://raiontech.io/blog/whatsapp-api-number-restricted-after-onboarding
- **Our gap:** we enforce opt-in but don't record opt-in *source/date* (Meta
  requires documented consent with source) and we have no opt-out line.

**Attribution & analytics:**
- Best flows attribute recovery to the step that caused it (per-step
  conversion), enabling A/B testing of timing/copy — Klaviyo's guidance is
  explicit: "Test out different offers and the timing of the sequence."
- **Our gap:** `recoveries` records no `recovered_via_step`; `stats()` has no
  per-step funnel; no revenue-per-recipient vs benchmark.

**Reliability:**
- Failed sends currently die as "failed" forever. Production practice:
  bounded retry with backoff (a bridge flapping for 5 minutes shouldn't cost
  the whole sequence).
- Abandoned carts that never convert should age into an "expired" state for
  honest funnel reporting (abandoned → recovered | expired).

**Medusa-native reference (luluhoc/medusa-plugin-abandoned-cart):**
- Subscriber pattern on cart events with `abandoned_count` on the cart and a
  `transformCart` normalization step. Validates our subscriber-based webhook
  design and our normalize-then-intake pipeline shape.

**The gold we lack — cart_recovery:**
1. Step 2 = objection handling (shipping/returns/trust), not a bare repeat.
2. Step 3 = optional real incentive (coupon code rendered in the message),
   reserved at intake, never in steps 1-2.
3. Per-step recovery attribution (`recovered_via_step`) + funnel stats
   (recovery rate, revenue per recipient vs the $3.65 Klaviyo benchmark).
4. Opt-in source/date recording + opt-out line on marketing steps.
5. Bounded retry (2 attempts, 15-min backoff) for failed sends.
6. UTM-tagged checkout links for step-level attribution.
7. `mark_expired()` + expired status for honest funnels.
8. Configurable sequence timings (keep 30m/24h/72h defaults — WhatsApp-fast).

---

## 2. `StoreManager` / Medusa provisioning

### How the best do it

**Medusa v2 admin auth (verified against real adapters):**
- Admin JWT: **`POST /auth/user/emailpass`** — NOT `/admin/auth`.
  Customer JWT: `POST /auth/customer/emailpass`. Store requests carry
  `x-publishable-api-key`.
  https://github.com/vishkaty/commerce-agents-medusa/blob/HEAD/docs/medusa-notes.md
  (mined against Medusa 2.20.1, 2026-09-04)
- The hermes medusa-admin skill confirms: pre-generated Admin API token is
  preferred (`Authorization: Bearer`), JWT via email/password is the fallback.
  https://github.com/gumbyender/hermes-skill-framework/blob/HEAD/packs/medusa/skills/medusa-admin/SKILL.md

**API keys — two different key types (our code conflates them):**
- **Publishable** keys: `POST /admin/publishable-api-keys` → returns
  `{publishable_api_key: {id, token}}`. The key **must be linked to a sales
  channel** (`POST /admin/api-keys/{id}/sales-channels`) or Store API product
  listings come back empty. Docs:
  https://docs.medusajs.com/api/admin (publishable-api-keys routes)
- **Secret/admin** keys: `POST /admin/api-keys` (creates admin tokens).
- **Our gap:** we POST `/admin/api-keys` and treat the result as publishable.
  Wrong type, and we never link a sales channel → storefront would list no
  products.

**Webhooks — no such REST route (our code invents one):**
- Medusa v2 has **no `/admin/webhooks` endpoint**. Outbound webhooks are wired
  via **subscribers**: `src/subscribers/<name>.ts` with
  `export const config = { event: "cart.updated" }`, the handler POSTing to
  the external URL. Custom inbound routes go under `src/api/`.
  https://github.com/greedychipmunk/medusa-plugin-printify/blob/HEAD/AGENTS.md
  https://github.com/luluhoc/medusa-plugin-abandoned-cart
- The v1 `medusa-plugin-webhooks` hackathon plugin is the "trash" lesson: it
  only supported 4 hardcoded events and died with v2 — don't bolt on a fake
  route; emit the real subscriber file.
- **Our gap:** step (d) POSTs to an invented `/admin/webhooks` and marks the
  store "ready" on a 200 from a route that cannot exist. Gold: generate the
  real subscriber TypeScript, store it in the registry for the operator to
  install, and report it honestly.

**Product creation (real v2 shape):**
- `POST /admin/products` takes `title`, `description`, `options`, and
  `variants: [{title, manage_inventory, prices: [{amount, currency_code}]}]`.
  (v2 docs: medusa.admin.products.createVariant with `prices: [{amount,
  currency_code}]`; amounts in smallest currency unit.)
  https://github.com/trivikr/medusa/blob/HEAD/www/apps/docs/content/modules/products/admin/manage-products.mdx
  https://github.com/oonid/toko-rs (17-endpoint product catalog, both create
  and update are POST)
- **Shipping-profile gotcha (vishkaty, verified):** a product created
  without `shipping_profile_id` cannot be checked out — cart completion fails
  even though the product exists. Must fetch the default profile
  (`GET /admin/shipping-profiles`) and set it on create.
- **Stock:** `variants.inventory_quantity` + `manage_inventory` are
  variant-level; update via `POST /admin/products/:id/variants/:vid`.
- **Our gap:** we POST a top-level `prices` list (wrong shape), never set a
  shipping profile (created products would fail checkout — a live bug), and
  never ask the LLM for an image spec despite `ProductDraft.image_spec`.

**Promotions (real v2 shape):**
- `POST /admin/promotions` with `{code, type, status, application_method:
  {type: "percentage", target_type: "order", allocation: "each", value,
  max_quantity, currency_code}}`. Automatic promotions require
  `application_method.max_quantity` with allocation `each`.
  https://github.com/vishkaty/commerce-agents-medusa/blob/HEAD/docs/medusa-notes.md
  https://github.com/hectasquareuk/medusa-plugin-variant-promotions
- `POST /admin/campaigns` holds spend budgets (campaigns ≠ marketing
  campaigns in the merchant sense).
- **Our gap:** we POST `{code, type: "percentage", value}` — the value lives
  at the wrong level and `application_method` (the required discount spec)
  is missing.

**WooCommerce (real REST API):**
- Webhooks: `POST /wp-json/wc/v3/webhooks` with
  `{name, topic, delivery_url, secret}`, basic auth
  `-u consumer_key:consumer_secret`.
  https://github.com/woocommerce/woocommerce-rest-api-docs/blob/HEAD/source/includes/wp-api-v3/_webhooks.md
- **Built-in topics cover orders/products/customers/coupons only — there is
  NO cart topic in WC core.** The documented pattern for cart signals is a
  custom action topic: `topic: "action.woocommerce_add_to_cart"`.
  https://github.com/agentnxt/agentskills (ecommerce-integration-expert)
- Webhook verification: HMAC-SHA256 of raw body, base64, compared against
  `X-WC-Webhook-Signature`; WC auto-disables webhooks after 5 consecutive
  delivery failures.
- **Our gap:** we register no webhook at all on the Woo path, and `_woo_auth`
  returns `{}` (no auth) — the check would fail against any real site with
  protected API.

**AI catalog generation:**
- Best practice: structured JSON schema from the LLM, strict validation
  (skip invalid rows, fail closed when zero valid), price sanity bounds.
  We do this; the gap is the wrong product payload shape (fixed above) and
  the missing image-spec plumbing.

**The gold we lack — medusa:**
1. Auth path `/auth/user/emailpass` (v2-correct).
2. Publishable key via `/admin/publishable-api-keys` + sales-channel link
   (`POST /admin/api-keys/{id}/sales-channels`).
3. Webhook step (d) = generate real subscriber TS (honest, installable),
   stored in registry; never mark "webhook configured" on an invented route.
4. Product payload: variants with inline `prices` (kobo), `manage_inventory`,
   `shipping_profile_id` fetched from the backend (checkout would otherwise
   fail), LLM-drafted image spec plumbed into product metadata.
5. Promotion payload with `application_method` (the required v2 shape).
6. Stock update via documented variant-update route.
7. WooCommerce: basic-auth header (from vault-resolvable refs or explicit
   creds), webhook registration with the documented custom cart topic
   (`action.woocommerce_add_to_cart`), delivery secret; remove the
   `FAILED if False` dead code.
8. `list()` N+1 fix (build from rows, don't re-query per store).
9. A shared `create_discount()` helper so the cart-recovery incentive step can
   mint a REAL coupon code via the store's own API.

---

## 3. What NOT to mine (trash lessons)

- v1-era Medusa blog posts (`/admin/auth`, `medusa-plugin-webhooks` with 4
  hardcoded events): the v2 API surface moved; trusting them is how the
  invented endpoints got in.
- Klaviyo "always include a discount" templates: trains abandon-for-discount
  behavior; discount only in the final nudge, high-value carts.
- WhatsApp tutorials that skip opt-in/template rules: the number gets banned;
  our opt-in gate is the compliance core — keep it, and record the source.
