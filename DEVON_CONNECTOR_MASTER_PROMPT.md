# DEVON CONNECTOR EXPANSION — MASTER BUILD PROMPT

> Copy everything below this line into your coding agent. It is self-contained.
> Target repo: https://github.com/Oluwacutyp/No-morals-ai · branch `arena/01a088e0-no-morals-ai`
> Existing pattern to follow: `nomorals/tools/github.py` (actions-based tool) + `tests/test_github_tool.py`.

---

## 0. CONTEXT

Devon is an all-purpose personal + coding agent. It has a tool registry (`nomorals/tools/`) where each
tool exposes named actions. You are adding a **unified connector system**: one framework + four
production connectors (finance, virtual cards, proxy pool, Nigerian commerce). Every connector must
support **multiple connection methods**, never just one — the owner refuses to be limited to a single way.

## 1. OBJECTIVE

Build, in this order:

1. **A. Connector framework** — `BaseConnector`, encrypted credential vault, 10 connection patterns.
2. **B. Finance connector** — Mono (Nigerian banks) + Plaid (US/EU banks) behind one unified interface.
3. **C. Virtual cards connector** — Privacy.com API (issue / pause / close / list virtual cards).
4. **D. Proxy pool** — multi-source proxy scraper + async validator + scoring + rotation.
5. **E. Nigerian commerce** — Jumia / Konga / Jiji / AliExpress / Temu via browser automation + a "real steal buys" deal hunter.

Each ships with its own test file. No real credentials in tests — mock all HTTP.

## 2. PART A — CONNECTOR FRAMEWORK

### 2.1 Files
- `nomorals/connectors/__init__.py`
- `nomorals/connectors/base.py` — `BaseConnector` abstract class
- `nomorals/connectors/vault.py` — encrypted credential vault
- `nomorals/connectors/patterns.py` — the 10 connection-pattern helpers

### 2.2 `BaseConnector` interface (every connector implements this)
```python
class BaseConnector(ABC):
    name: str                 # e.g. "mono", "privacy_cards"
    description: str
    @abstractmethod
    def status(self) -> dict: ...
        # -> {"connected": bool, "account": str|None, "scopes": [...], "last_check": iso}
        # MUST check live every call. Never cache across calls, never trust memory.
    @abstractmethod
    def connect_url(self) -> str: ...
        # Exact URL to show the user. Never invent one; return "" if none applies.
    @abstractmethod
    def disconnect(self) -> dict: ...
    def refresh(self) -> dict: ...   # token refresh; default: {"refreshed": False, "reason": "n/a"}
    def capabilities(self) -> list[str]: ...
        # ONLY what is actually implemented. Scope honesty is a hard rule:
        # connecting Spotify-like service X does NOT imply unlisted features.
```

### 2.3 Credential vault (`vault.py`)
- Encrypted at rest (Fernet, key from `NM_VAULT_KEY` env or OS keyring; generate on first run).
- API: `vault.store(connector, label, secret_dict)`, `vault.get(connector, label)`,
  `vault.delete(connector, label)`, `vault.list_labels(connector)` (labels only, never values).
- **Hard rules:** secrets never in logs, never in tool output, never in memory files, never in
  exception messages. Card PANs/CVVs are masked (`****1234`) everywhere except the single
  checkout handoff that needs them.

### 2.4 The 10 connection patterns (`patterns.py`)
Implement helpers for each; every new connector picks ≥2:
1. `oauth_accounts_center` — auth URL → user signs in at provider → callback stores tokens.
2. `oauth_provider_hosted` — provider's own authorize page; `authorize_url()` →
   `exchange_code(code)` → `refresh()`. (Model on Notion/Mono-style flows.)
3. `session_link` — piggyback an existing platform session (cookies/token bridge).
4. `consent_flow` — permission grant with no sign-in (public/read-only data).
5. `api_key_vault` — **generic**: hosted secure page takes ANY service's API key → vault.
   This is the universal fallback. Build it once, reuse for everything.
6. `password_vault_browser` — saved login + automated browser drives the real website.
   (This is how services with NO api get connected.)
7. `direct_protocol` — IMAP/SMTP/CalDAV style: host + port + app password from user.
8. `device_local` — paired-device / desktop-app bridge, no cloud auth.
9. `mcp_server` — consume OAuth-backed MCP servers; expose Devon tools as MCP too.
10. `browser_fallback` — signed-in browser session as the connector of last resort.

## 3. PART B — FINANCE CONNECTOR (Mono + Plaid)

### 3.1 Files
- `nomorals/connectors/finance.py` — `MonoConnector`, `PlaidConnector`, unified `Finance` facade
- `nomorals/tools/finance.py` — tool actions: `link`, `accounts`, `transactions`, `balance`, `identity`, `unlink`, `status`
- `tests/test_finance_connector.py`

### 3.2 Mono (Nigerian banks — GTB, Access, FirstBank, UBA, Zenith, …)
- Base URL: `https://api.withmono.com/v2` (confirm in current docs before shipping)
- Auth header: `mono-sec-key: <secret key>` (`test_sk_…` sandbox / `live_sk_…` production, from vault)
- Linking: embed Mono Connect widget (`https://connect.mono.co/connect.js`) with the **public** key;
  `onSuccess` returns a temporary `code` → server exchanges it:
  `POST /v2/account/auth` with `{"code": code}` → returns `id` (account id). Store the id.
- Data endpoints (confirm paths in current docs):
  `GET /v2/accounts/{id}`, `GET /v2/accounts/{id}/transactions`,
  `GET /v2/accounts/{id}/identity`, `GET /v2/accounts/{id}/income`,
  `POST /v2/accounts/{id}/unlink`
- Webhooks: verify HMAC-SHA512 using `mono-webhook-secret` header before trusting any event.

### 3.3 Plaid (US/EU banks)
- Standard flow: `POST /link/token/create` → user links → `public_token` →
  `POST /item/public_token/exchange` → `access_token` (vault).
- `POST /accounts/balance/get`, `POST /transactions/sync`, `POST /identity/get`.
- Envs: sandbox / production. **Plaid does NOT support Nigerian banks** — route NG users to Mono.

### 3.4 Unified facade
`Finance(provider="auto")` picks Mono for NG-linked accounts, Plaid otherwise.
Actions: `finance.link(provider)`, `finance.accounts()`, `finance.transactions(account_id, days=30)`,
`finance.balance()`, `finance.identity()`, `finance.unlink(account_id)`.
All money amounts normalized to minor units + currency code.

## 4. PART C — VIRTUAL CARDS (Privacy.com)

> NOTE: Privacy.com is US-only (requires a US bank account). If the owner is non-US, implement the
> same interface against **Stripe Issuing** as fallback. Interface first, provider second.

### 4.1 Files
- `nomorals/connectors/cards.py` — `PrivacyCardsConnector`
- `nomorals/tools/cards.py` — actions: `create`, `list`, `get`, `pause`, `unpause`, `close`, `transactions`
- `tests/test_cards_connector.py`

### 4.2 API (verified 2026-09-29 — re-confirm before shipping)
- Base: `https://api.privacy.com/v1` · Auth: `Authorization: api-key <KEY>` (vault) · Amounts in **cents**.
- `POST /v1/cards` → `{type, memo, spend_limit, spend_limit_duration, state}`.
  Types: `SINGLE_USE` (auto-closes after one charge), `MERCHANT_LOCKED` (locks to first merchant),
  `DIGITAL_WALLET`. Durations: `TRANSACTION`, `MONTHLY`, `ANNUALLY`, `FOREVER`.
- `PATCH /v1/cards/{token}` → update `state` (`PAUSED`/`CLOSED` — CLOSED is permanent), `spend_limit`, `memo`.
- `GET /v1/cards`, `GET /v1/cards/{token}`, `GET /v1/transactions` (paginate).
- Response includes `pan`, `cvv`, `exp_month`, `exp_year`, `token`, `last_four` — mask PAN/CVV in all
  storage and logs; reveal full PAN only inside the single checkout handoff.

### 4.3 Behaviors
- `cards.create_for_purchase(merchant, amount_cents)` → SINGLE_USE card, limit = amount rounded up
  to whole dollars, memo = merchant. Auto-closes after charge.
- `cards.create_for_subscription(merchant, monthly_cents)` → MERCHANT_LOCKED, MONTHLY limit.
- Never invent card numbers. All cards come from the provider API.

## 5. PART D — PROXY POOL (scraper + validator + rotation)

### 5.1 Files
- `nomorals/connectors/proxies.py` — `ProxyPool`
- `nomorals/tools/proxies.py` — actions: `scrape`, `validate`, `list`, `best`, `remove`, `stats`
- `tests/test_proxy_pool.py`

### 5.2 Real sources (all free, no key needed unless noted)
1. ProxyScrape v2: `https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all`
   (also `protocol=socks4`, `socks5`)
2. GeoNode: `https://proxylist.geonode.com/api/proxy-list?limit=500&page=1&sort_by=lastChecked&sort_type=desc`
   (filterable: country, anonymityLevel, protocols)
3. `https://www.proxy-list.download/api/v1/get?type=https` (types: http/https/socks4/socks5)
4. `https://free-proxy-list.net` (HTML scrape — elite/anonymous/transparent labeled)
5. WebShare: 10 free proxies with account (API key in vault)

### 5.3 Pipeline
- **Scrape** (scheduled, e.g. every 6h): pull all sources → normalize to
  `{ip, port, protocol, country, anonymity, source}` → dedupe → SQLite table `proxies`.
- **Validate** (async, concurrent): request `https://httpbin.org/ip` through each proxy;
  record `latency_ms`, `working bool`, `anonymity_real` (compare returned IP:
  elite = no proxy headers + IP hidden, anonymous = headers present, transparent = IP leaked).
  Kill anything with latency > 3000ms or failing.
- **Scoring**: `score = w1*speed + w2*uptime_ratio + w3*anonymity_weight - w4*fail_streak`.
  **Decay**: re-validate on schedule; dead proxies lose score fast and auto-prune below threshold.
- **Rotation**: `proxies.best(country=None, protocol=None)` returns highest-scored working proxy
  and marks it in-use; on request failure the caller reports back and the proxy is penalized instantly.
- **Stats**: counts by country/protocol/anonymity, avg latency, pool health.

## 6. PART E — NIGERIAN COMMERCE (browser automation + deal hunter)

### 6.1 Files
- `nomorals/connectors/commerce_ng.py` — site adapters
- `nomorals/tools/deals.py` — actions: `search`, `steals` (the "real steal buys" hunter), `track_price`, `alerts`
- `tests/test_deals.py`

### 6.2 Adapters (all via connection pattern #6 / #10 — no public APIs)
Jumia, Konga, Jiji, AliExpress, Temu. Each adapter implements:
`search(query) -> [ {title, price_ngn, url, seller, rating} ]`,
`product(url) -> detail`, `price_history(url)`.
Currency normalize everything to NGN.

### 6.3 "Real steal buys" hunter
- `deals.steals(query, max_price_ngn)` searches all five sites, flags listings priced
  significantly below the median for equivalent items (configurable threshold, default 35% below),
  filters obvious scams (no seller history + too-good price), ranks by discount %.
- `deals.track_price(url)` stores price snapshots; scheduled check alerts the owner on drops.
- Polite scraping: rate-limit per site, rotate user agents, use the proxy pool from Part D.

## 7. WIRING INTO DEVON
- Register each tool in the tool registry following the existing pattern
  (see `nomorals/tools/github.py` + `tests/test_github_tool.py`).
- CLI: `nm connectors list`, `nm connectors connect <name>`, `nm connectors status`.
- Config: per-connector settings dataclass + `NM_`-prefixed env mapping (single prefix —
  the double-`NM_NM_` bug must not be repeated).
- DB migrations for new tables: `proxies`, `proxy_checks`, `cards`, `finance_accounts`,
  `price_watches`, `connector_credentials` (labels only — secrets stay in the vault).

## 8. TESTS & ACCEPTANCE CRITERIA
- One test file per part (B–E), all HTTP mocked, zero real credentials.
- Framework tests: vault encrypt/decrypt round-trip, vault never leaks values in
  `list_labels()` or exceptions; `status()` called live twice returns fresh results.
- Finance: Mono auth-code exchange mocked → account id stored; Plaid link-token flow mocked.
- Cards: create/pause/close mocked; assert PAN masked in stored records.
- Proxies: scraper parses each source format; validator scoring math unit-tested;
  rotation penalizes a reported failure.
- Deals: steal-detection flags a 40%-below-median listing; scam filter rejects no-history sellers.

## 9. SECURITY RULES (non-negotiable)
- Secrets in the vault only. Never in code, logs, tool output, memory files, or exceptions.
- Webhook signatures verified (Mono HMAC-SHA512) before acting on events.
- No self-generated card numbers — provider APIs only.
- Scraping stays polite: rate limits, proxy rotation, respect robots.txt where feasible.

## 10. NON-GOALS
- No offensive attack tooling. No password-cracking improvements.
- No inventing connect URLs — if a provider has no URL flow, say so and use the vault/browser patterns.
- Do not claim a connector supports a capability its provider doesn't document.

## 11. DELIVERABLE
Working code + tests passing (`python -m pytest tests/test_finance_connector.py
tests/test_cards_connector.py tests/test_proxy_pool.py tests/test_deals.py -q`),
plus a short `CONNECTORS.md` in the repo root documenting each connector, its connection
methods, and its actions. Do NOT commit or push — leave changes in the working tree.
