# INTEGRATIONS Sweep — External Mining Report

Module: `nomorals/integrations/` — 15 files (calendar, email, shopping x3,
smarthome, mqtt, digital_twin, routines, market_data, sentinel_bridge,
payment, stt, voice).

Mined 2026-10-10. Each class below gets: what the BEST outside
implementation does, what OUR code lacks, and what gets built.
Style/UX is mined too — "how should this LOOK/FEEL when the user
interacts with it?"

---

## 1. CalendarIntegration (calendar_integration.py)

**Best outside:**
- `kuzmoyev/google-calendar-simple-api` (gcsa): pythonic Event objects,
  `Recurrence.rule(freq=DAILY)` first-class, `minutes_before_popup_reminder`
  sugar, iterator protocol over events, beautiful-date integration.
- `gcal-sync` (PyPI): async generator paging (`ListEventsRequest` +
  async iteration), incremental sync via syncToken, local store with
  recurrence expansion at query time (`CalendarEventSyncManager`).
- lemma-ai calendar sync: pluggable provider ABC + registry,
  Google (OAuth + incremental syncToken) + Apple CalDAV (iCloud,
  app-specific password), sync engine with conflict detection.
- Best practice (Google docs): never full re-list — persist
  `nextSyncToken`, handle 410 Gone (token expired → full re-sync),
  `sendUpdates=none` to avoid spamming attendees on programmatic writes,
  batch endpoint for multi-op writes, `timeMin/timeMax` + `singleEvents`
  + `orderBy=startTime` for agenda views.

**Our gaps:** Google-only, no CalDAV, no token refresh (dies when the
access token expires — no refresh_token flow at all), no recurrence
support, no freebusy lookup, no search, no `list_calendars`,
events returned raw without pretty rendering, UTC-only timezone,
`datetime.utcnow()` (deprecated).

**Build:** token refresh flow (refresh_token → new access_token,
persisted back into the vault); `CalendarProvider` ABC so CalDAV can
plug in later without a rewrite; `add_event` recurrence param
(RRULE); `find_free_time()` via freebusy API; `search_events()`;
`list_calendars()`; pretty `format_agenda()` day rendering with
emoji/time grouping; `sync_token` incremental sync storage in sqlite;
`send_updates` flag defaulting to "none" for agent-driven writes;
per-account timezone instead of hardcoded UTC.

---

## 2. HomeTwin (digital_twin.py)

**Best outside:**
- Eclipse Ditto: Thing = entity + Features (reported AND desired state),
  policies (who may read/write), Things-Search (query across all twins),
  connectivity adapters (MQTT/AMQP/HTTP), Ditto Protocol messages.
  Key insight: a twin is not a log — it's **reported vs desired state**
  with command routing back to the device.
- Smart-elevator twin (abdelrahmen-zaouidi): deterministic **safety
  gate** before commands flow back (risk score, cooldowns,
  twin-freshness checks), TimescaleDB telemetry + command audit log.
- Digital-twin-project (classroom): Telegraf/InfluxDB timeseries +
  Grafana; Node-RED flows; Unity 3D visualization.

**Our gaps:** no desired-vs-reported state (write-back impossible),
no per-entity metadata registry (device class, unit, area), no
prediction ("when will the washing machine finish?"), no energy rollup
by domain/area, no retention policy (DB grows forever), no export,
`summary()` is plain bullet text — no sections, no severity ordering,
no "good morning" briefing render.

**Build:** `DesiredState` table + `set_desired()`/`desired_state()`
(report vs command intent, Ditto-style); entity metadata registry
(device_class, unit, area) auto-enriched from HA attributes;
retention (`prune(days)`) with WAL mode; `energy_today()` rollup for
power sensors; `predict()` stubs done honestly as linear extrapolation
on numeric sensors ("on current trend, X in ~Y min") — statistical,
never claimed intelligent; `format_briefing()` — god-tier morning
briefing render (sections, severity-ordered anomalies, rhythm notes);
JSON export/import.

---

## 3. EmailIntegration (email_integration.py)

**Best outside:**
- `imap-tools` / `IMAPClient`: UID-based fetch, `use_uid=True`,
  server-side SEARCH criteria (not client filtering), `BODY.PEEK` to
  avoid marking read, batch fetch.
- Gmail best practice (multiple production guides): **History API
  over Search** — `users.history.list(startHistoryId)` returns only
  what changed; watch expiry is 7 days → renew on schedule; push
  messages carry historyId but are NOT the message (fetch via history);
  deduplicate on historyId; `sendUpdates`/`sendAs` for identity.
- `chrnvpy/email-agent`: OAuth once (`gmail.modify` scope), token.json
  auto-refresh, Pub/Sub push with JWT verification.

**Our gaps:** no History API incremental sync (always full list —
slow + quota burn), no `watch()`/push support, no threads API, no
drafts, no batch modify (mark many read in one call), no attachment
download, no `list_labels`, message body extraction is shallow
(snippet only), no digest/triage rendering for chat ("📬 3 unread"
with sender/subject/action lines).

**Build:** `history_id` cursor per account in sqlite + `sync_new()`
(history.list incremental); `start_watch(topic)`/`stop_watch()` with
expiry tracking + `renew_watches()` helper; `get_thread()`,
`create_draft()`/`send_draft()`; `batch_modify()` (labels add/remove
many at once); `download_attachment()`; `format_digest()` god-tier
inbox rendering (grouped by thread, importance hints, one-line
actions); HTML→text body extraction upgrade (readability-style);
`search` keeps working but sync_new is the default path.

---

## 4. market_data.py (functions + SentinelMarketProvider)

**Best outside:**
- `ccxt` (35k stars): THE unified API — 100+ exchanges, one method
  shape (`fetch_ohlcv`, `fetch_ticker`, `fetch_order_book`,
  `fetch_trades`), `enableRateLimit=True` built-in throttle,
  `exchange.has` capability checks, sandbox mode, `ccxt.pro`
  WebSocket-first (`watch_ohlcv`, `watch_ticker` — no REST rate-limit
  burn), `load_markets()` once + cache, `since`-pagination for deep
  history.
- Our file hand-rolls every exchange adapter with bespoke parsing —
  exactly what ccxt exists to kill.

**Our gaps:** no unified adapter (each source is custom code),
no WebSocket streaming at all, no TTL cache (every call hits
network), no order-book / bid-ask spread (execution honesty),
no `since`-pagination (bars capped by one request), no indicator
helpers (RSI/EMA/MACD — every consumer re-implements),
no multi-symbol batch quotes, quote dict is thin (no bid/ask/high/low).

**Build:** ccxt-backed `CCXTSource` adapter (optional dep, graceful
fallback to existing fetchers): `fetch_ohlcv` unified, `enableRateLimit`,
`load_markets` cached; TTL cache layer (`@ttl_cache`) for quotes and
OHLCV; `order_book()` returning bid/ask/spread; `trades()` tape;
`indicators(df)` — RSI/EMA/MACD/ATR/Bollinger in pure python+numpy-free
(stdlib fallback, pandas-aware when present); `batch_quotes()`;
`stream_quotes()` async generator over ccxt.pro `watch_ticker` when
available with REST-poll fallback; richer quote dict
(bid/ask/high/low/volume). Keep ALL existing keyless fetchers as the
offline fallback chain — ccxt is an upgrade path, never a hard dep.

---

## 5. MQTTBridge (mqtt_client.py)

**Best outside:**
- paho-mqtt canon: `loop_start()` handles reconnect automatically;
  manual `loop()` requires hand-rolled reconnect with backoff
  (steves-internet-guide pattern: retry_delay doubling to 100s cap,
  quit-on-auth-failure); unique client IDs; LWT (last will) +
  birth messages; retained discovery for HA MQTT discovery;
  resubscribe inside `on_connect` (subscriptions die with the session
  on clean reconnect).
- Zigbee2MQTT contract: `zigbee2mqtt/<device>` state topic JSON,
  `.../set` command topic, `.../get` poll topic, `.../availability`,
  bridge state topic `zigbee2mqtt/bridge/state`.

**Our gaps:** no LWT/birth messages, no retained publish option,
no TLS config surface, no resubscribe-on-reconnect audit,
no availability tracking wired to twin, no per-topic message-rate
stats, `set_state` returns bool but doesn't confirm via state echo,
no HA MQTT discovery publish helper (for Devon-originated virtual
devices).

**Build:** LWT + birth/online messages; `publish(..., retain=...)`;
TLS options passthrough; automatic resubscribe on reconnect (already
partially there — harden + log); `availability` → twin offline
ingest; `stats()` (msgs/s per topic, uptime, reconnect count);
`publish_discovery()` — HA MQTT discovery payload builder for
virtual devices Devon creates; `wait_for_state()` — publish then
await echo with timeout (honest command confirmation).

---

## 6. NaijaDealHunter + NaijaShoppingEngine (naija_deals.py, naija_shopping.py)

**Best outside:**
- camelcamelcamel / Keepa: price HISTORY charts are the product, not
  the price — "is this actually a deal?" needs the 90-day curve;
  price-drop alerts with threshold + cooldown; per-product watchlists.
- Honey: coupon/price comparison at point of decision.
- Best deal UX: deal cards with score badges, was/now/%-off, "lowest
  in 90 days" flags, urgency without dark patterns.

**Our gaps:** `_get_price` scrapes are single-point (no history
curve), deal score exists but no "lowest-ever" flag, no coupon/promo
detection, no stock/availability tracking, `to_message()` cards are
plain, no deal digest scheduling hook, no multi-vendor "same product"
matching (title normalization exists in shopping but not wired to
deals), no FX display toggle (₦/USD).

**Build:** price-history curve stored per product + `is_lowest_90d`
flag on deals; coupon/promo-code extraction from product pages;
stock status tracking (in/out + "back in stock" alerts);
`to_message()` god-tier deal cards (score badge, was→now, % off,
lowest-ever flag, vendor trust line); `format_digest()` daily deals
briefing; FX toggle in rendering (₦ default, USD secondary);
cross-vendor product matching (`match_product()` using existing
`_norm_title` + new similarity) so one product shows all vendor
prices side-by-side.

---

## 7. PaymentIntegration (payment_integration.py)

**Best outside:**
- web3.py: THE Ethereum python lib — `eth.get_balance` (wei→ether
  via `from_wei`), contract calls via ABI, event listening, signed
  sends. ERC-20 balance = `contract.functions.balanceOf(addr).call()`
  with the standard minimal ABI.
- Blockstream Esplora / blockbook: address-level reads (balance,
  tx history, UTXOs) without running a node — exactly our balance
  use-case, more reliable than blockchain.info scraping.
- Production payment UX: approval flow with expiry (we have it),
  fee estimation display, QR receive codes, fiat equivalents
  everywhere, tx status tracking (pending → confirmed).

**Our gaps:** no ERC-20 token support (ETH-only on EVM), no gas
estimation shown pre-send, no tx confirmation tracking
(`track_tx()` poll → confirmed), no QR code for receive addresses,
fiat equivalents missing on send/receive, address validation is
regex-only (no checksum/EIP-55), no saved address book, no
per-currency explorer links in messages.

**Build:** ERC-20 balance + transfer via minimal ABI (web3 optional
dep, graceful without); `estimate_fee()` surfaced in approval cards;
`track_tx(txid)` with confirmation polling + `format_tx_status()`;
QR code generation for receive addresses (qrcode lib optional,
ASCII fallback); fiat equivalents on every money message;
EIP-55 checksum validation; address book (named, sqlite);
explorer links per currency in transaction messages;
`format_wallet()` god-tier balance card (per-currency rows,
fiat totals, 24h sparkline hint via market_data.quote change %).

---

## 8. routines.py (NL → HA automation)

**Best outside:**
- HA best practices (derguru skill): entity_id over device_id,
  automation `mode` matters (single vs restart vs queued vs
  parallel), `subscribe_trigger` for server-side filtering,
  built-in helpers before templates, trigger IDs.
- AppDaemon / Node-RED: the gold standard is a VISUAL + LLM hybrid —
  Node-RED flows show that routines want branching (choose), delays,
  waits, and loops, not just flat action lists.
- Modern LLM routine builders: conversational repair ("which kitchen
  light — the main or the island?") instead of fail-and-retry.

**Our gaps:** no `mode` selection (HA defaults single — wrong for
motion lights), no wait/delay actions ("turn off after 10 min"),
no choose/branching, no numeric-state conditions ("only if temp >
28°"), no trigger duration ("no motion for 5 min"), scene action
doesn't resolve scene names to entity_ids, clarification is
one-shot text (no structured options), `describe()` is one
sentence — no step breakdown.

**Build:** `mode` param on Routine (single/restart/queued/parallel,
smart default by starter kind); `wait`/`delay` action parsing
("then wait 5 minutes", "turn off after 10 minutes") → HA `delay`
+ `wait_template`; numeric-state conditions ("only if {sensor}
above/below {n}"); trigger `for:` duration ("when no motion for 5
min"); scene name → entity_id resolution against device list;
`describe()` god-tier: numbered step breakdown with trigger,
conditions, actions sections; `suggest_fix()` — structured repair
options for each validation error (not just text); LLM fallback
hook (`llm_parse` callable) when regex parse fails — pluggable,
never required.

---

## 9. sentinel_bridge.py (Sentinel.py trading bridge)

**Best outside:**
- freqtrade (28k stars, 500+ contributors): the reference —
  strategy class with `populate_indicators`/`populate_entry_trend`,
  `minimal_roi` ladder, trailing stoploss, hyperopt
  (`SortinoHyperOptLossDaily`), walk-forward, **dry-run paper
  trading**, `--enable-protections` (cooldown, stoploss guard,
  max drawdown), Telegram `/stop` emergency brake, graduated
  capital deployment (10% → 25% → 50% → 100%).
- Voltra: honest pipeline — backtest → walk-forward → Monte Carlo
  (P(edge>0) ≥ 95% bar) → 30-day dry-run → small live. Never skip.
- Doctor pattern: pre-flight checks before live.

**Our gaps:** no backtesting harness at all (strategies can't be
validated), no paper-trading mode, no risk guardrails (daily
stop-loss, max drawdown kill-switch), no fee/slippage modeling,
no performance report rendering, `doctor()` exists — good —
but no strategy registry/metadata.

**Build:** `backtest(strategy, bars, fee, slippage)` — vectorized
backtester returning trades/equity curve (honest, fee+slippage
modeled); `paper_trade()` dry-run engine on live data;
`RiskGuard` — daily stop-loss %, max drawdown %, max open trades,
cooldown after N consecutive losses, emergency `halt()`; strategy
registry with metadata (name, description, params, backtest
summary); `format_report()` god-tier backtest card (win rate,
profit factor, max drawdown, Sharpe-ish, equity sparkline as
ASCII); Monte Carlo trade reshuffle for edge significance
(P(profit>0)); graduated capital sizing helper. Keep the real
Sentinel engine as the execution path — this is the validation
layer it lacks.

---

## 10. ShoppingIntegration (shopping_integration.py)

**Best outside:**
- Production shopping agents: review sentiment extraction, seller
  ratings, shipping ETA + cost, price-history ("was $X 30 days ago"),
  variant selection (size/color), cart persistence across sessions.

**Our gaps:** no review/rating extraction, no shipping info, no
seller reputation, products lack images in rendering,
`PriceComparison` is text-flat, no wishlist persistence, search
results have no dedupe across retailers.

**Build:** rating + review-count extraction per product; shipping
ETA/cost parsing where available; `to_card()` rich product
rendering (price, rating stars, review count, shipping line,
image URL carried through); `PriceComparison.format()` god-tier
side-by-side table (best price highlighted, savings vs max);
wishlist (sqlite, price-drop alerts hook into naija check_alerts
pattern); cross-retailer dedupe by normalized title; `sort_by`
(price/rating/deals) on search.

---

## 11. SmartHomeIntegration + HAWebSocket (smarthome_integration.py)

**Best outside:**
- `balloob/llm-skills` python_api: **`subscribe_trigger` is the
  PREFERRED method** — let HA's automation engine filter
  server-side (state/numeric_state/time_pattern/template triggers)
  instead of streaming all state_changed and filtering client-side.
- ha best practices: REST for commands + WebSocket for state
  (hybrid), `?return_response=true` service calls, `/api/template`
  server-side rendering, bulk service calls with multiple entity_ids,
  minimal subscriptions, exponential-backoff reconnect.
- Our HAWebSocket is hand-rolled RFC 6455 over asyncio streams —
  impressive, zero-dep, keep it.

**Our gaps:** no `subscribe_trigger` (we stream everything and
filter client-side — wasteful); no service-call-with-response;
no template rendering; no areas/floors/labels; no device registry;
no media_player browsing; no camera snapshot; no `toggle`;
no light effects/color modes; scenes are list-only (no create with
entities snapshot); `Device`/`DeviceState` rendering is raw dicts.

**Build:** `subscribe_trigger(trigger, callback)` on HAWebSocket
(server-side filtering — the single biggest efficiency win);
`call_service(..., return_response=True)`; `render_template()`;
areas/floors/labels listing + `devices_in_area()`; device registry
query; `toggle()`; light effects + `color_temp_kelvin`/`hs_color`;
`capture_scene()` (snapshot current states → scene.create);
camera `snapshot()`; media_player `browse_media()` passthrough;
`format_devices()` god-tier room-by-room rendering with status
emoji; bulk `turn_on(entity_ids=[...])`.

---

## 12. SpeechToText (stt.py) + voice_integration.py (TTSEngine/STT)

**Best outside:**
- faster-whisper canon: `device="cpu", compute_type="int8"` for CPU;
  `vad_filter=True` (Silero VAD kills hallucinations in silence);
  `word_timestamps=True`; `condition_on_previous_text=False` on long
  form (prevents drift); `BatchedInferencePipeline` for long files;
  beam_size=5 standard; explicit `language=` to avoid mis-detection.
- WhisperX: faster-whisper + pyannote diarization → speaker labels
  + word timestamps in one pass.
- edge-tts: free, 100+ languages, `--rate/--volume/--pitch`,
  word-boundary subtitles (`--write-subtitles`), streaming via
  Communicate; hass-edge-tts, Podcastfy, openai-edge-tts (OpenAI-
  compatible endpoint + SSE streaming) prove the patterns.
- Piper/Kokoro: local neural TTS when offline matters.

**Our gaps (STT):** no VAD filtering, no word timestamps, no
diarization option, no language pinning, no model-size ladder
(tiny→large with auto-pick by hardware), no SRT/VTT export,
no hallucination guards (`condition_on_previous_text`,
`no_speech_threshold`), no streaming/chunked long-file handling.
**Our gaps (TTS):** no rate/volume/pitch control, no streaming,
no long-form chunking (2000-char edge limit needs splitting),
no voice preview/selection UX, no subtitle/word-boundary output,
no audio post-processing (loudness normalize).

**Build (STT):** faster-whisper backend with VAD + word timestamps
+ language pin + model ladder + `vad_filter`/`beam_size`/
`condition_on_previous_text` exposed; `transcribe_diarized()`
(WhisperX-style via optional dep, honest fallback message);
`to_srt()`/`to_vtt()` export; chunked long-file transcription;
hallucination guard defaults.
**Build (TTS):** rate/volume/pitch params; `synthesize_stream()`
async generator (edge-tts Communicate streaming); long-text
auto-chunker (sentence-aware, 2000-char chunks, crossfade-free
concat); voice catalog with preview text + `pick_voice()`
(language/gender/use-case filters); word-boundary events →
`to_srt()` subtitles; loudness normalize post-pass (optional);
`format_voices()` god-tier voice browser.

---

## Style/UX doctrine (applies to every class)

God-tier = **cards, not dumps**. Every user-facing render follows:
status emoji → headline → key facts → actions. Tables via a shared
`format_table()` helper (no new dep — stdlib). Money always shows
fiat equivalents. Times always show relative ("in 20 min") alongside
absolute. Errors suggest the fix (`suggest_fix()` pattern from
routines). No raw dicts ever reach chat.
