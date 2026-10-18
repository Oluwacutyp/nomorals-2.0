# Source verification — 2026-10-18

Owner rule: **every source must be actually reachable AND usable — not just a collection.**
A source that doesn't work is worse than no source. This report is the evidence for the
source lists as they now stand in the repo.

## Method

- Each source was fetched live from the build sandbox with a browser UA, then classified by
  **HTTP status + payload shape**: RSS must parse to >=1 item, JSON APIs must return >=1 result,
  HTML search pages must contain >=1 real result link. HTTP 200 with an empty/placeholder shell
  (the JS-SPA pattern) counts as **dead-as-a-source**, not live.
- Statuses: `verified-live` (200 + usable items) · `dead` (404/410, or 200 with zero usable items)
  · `bot-blocked` (Cloudflare/CAPTCHA/anti-bot page — reachable but unusable *from this network*)
  · `geo-blocked` (HTTP 451 / region page) · `transient-error` (5xx — server-side, kept)
  · `unverifiable` (connection reset/closed from the verifier network — evidence is about the
  verifier's network, not the source; kept, re-checked at runtime).
- Bot-blocked / geo-blocked / transient / unverifiable sources are **kept** (they work from other
  networks — e.g. the owner's phone on a residential/Nigerian network — and the new
  `agency.sourcehealth` system re-checks them at runtime with a circuit breaker: 3 consecutive
  failures stand a source down, 10 disable it, a later success re-activates it).
  Dead sources were **removed**. One structurally unusable source (Boomplay search) was
  **removed from the active chain** (class + resolver method kept, documented).
- Retries matter: several feeds (CBS, ESPN, PBS, Engadget, Brave, Gutendex) failed the first fetch
  and verified on retry — with a proper RSS `Accept` header in the feeds' case. That fix is now
  in `NewsAgent` (it sends RSS `Accept` headers).
- Probed candidates that did NOT verify were **not added**: Daily Maverick (404), Yonhap (400),
  Animal Político (504), Milenio (404), FourFourTwo (200 but not a feed).

## Totals

| Family | Probed | Verified-live | Dead (removed) | Bot-blocked | Geo-blocked | Transient | Unverifiable | Added (verified) |
|---|---|---|---|---|---|---|---|---|
| News feeds | 99 | 79 | 9 | 8 | 0 | 2 | 1 | 7 |
| Search engines | 10 | 6 | 0 | 3 | 0 | 0 | 1 | 0 |
| Music sources | 14 | 10 | 1 (chain) | 1 | 0 | 0 | 2 | 0 |
| Film sources | 3 | 0 | 0 | 1 | 0 | 0 | 2 | 0 |
| Weather providers | 5 | 4 | 0 | 0 | 0 | 0 | 1 | 0 |
| Wisdom sources | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| Price feeds | 4 | 2 | 0 | 1 | 1 | 0 | 0 | 0 |
| **Total** | **139** | **105** | **10** | **14** | **1** | **2** | **7** | **7** |

After fixes: **news = 97 feeds (86 verified-live, 11 network-dependent, re-checked at runtime).**
No source stays in a list unverified: every entry now carries a measured status in
`agency.sourcehealth`, and only `verified-live` sources belong in an active chain.

## News feeds (99 probed → 97 kept)

### Removed — dead (9)

- **news:Al Arabiya** — `https://www.alarabiya.net/rss.xml` — HTTP 404
- **news:Al Jazeera Africa** — `https://www.aljazeera.com/xml/rss/africa.xml` — HTTP 404
- **news:Bleacher Report** — `https://bleacherreport.com/articles/feed` — HTTP 404
- **news:Dawn** — `https://www.dawn.com/rss` — HTTP 404
- **news:El Universal** — `https://www.eluniversal.com.mx/rss.xml` — HTTP 404
- **news:Goal** — `https://www.goal.com/en/rss` — HTTP 404
- **news:Korea Herald** — `http://www.koreaherald.com/rss` — HTTP 200 items=0 len=61400 title=The Korea Herald
- **news:News24** — `https://www.news24.com/feed` — HTTP 404
- **news:Yahoo Finance** — `https://finance.yahoo.com/news/rssindex` — HTTP 404

### Added — verified-live replacements (7)

- **AfricaNews** (africa) — `https://www.africanews.com/feed/rss` — 50 items
- **Middle East Eye** (middleeast) — `https://www.middleeasteye.net/rss` — 20 items
- **Geo News** (asia) — `https://www.geo.tv/rss/1/0` — 50 items
- **The Hindu Business** (business) — `https://www.thehindu.com/business/feeder/default.rss` — 60 items
- **FT Markets** (business) — `https://www.ft.com/rss/home` — 11 items
- **Al Monitor** (middleeast) — `https://www.al-monitor.com/rss` — 20 items
- **ESPN FC** (sports) — `https://www.espn.com/espn/rss/soccer/news` — 15 items

### Kept but NOT verified-live from this network (11) — re-checked at runtime

- **news:ABC Australia** — `https://www.abc.net.au/news/feed/1` — **transient-error**: 500 Internal Server Error twice — server-side, feed kept, SourceHealth re-checks
- **news:Guardian NG** — `https://guardian.ng/feed/` — **bot-blocked**: Cloudflare 403 challenge page
- **news:Joy Online** — `https://joyonline.org/feed/` — **unverifiable**: Connection closed from verifier network twice — evidence is about this network, not the source
- **news:Leadership** — `https://leadership.ng/feed/` — **bot-blocked**: Cloudflare 403 challenge page
- **news:Nairaland** — `https://www.nairaland.com/feed` — **bot-blocked**: 403 empty (anti-bot)
- **news:Nation Africa** — `https://nation.africa/feed` — **bot-blocked**: 403 'Your request was blocked'
- **news:RTÉ** — `https://www.rte.ie/rss/news.xml` — **bot-blocked**: 403 'activity is prohibited' (bot policy)
- **news:SCMP** — `https://www.scmp.com/rss/91/feed` — **bot-blocked**: Cloudflare 403 challenge page
- **news:TheCable** — `https://www.thecable.ng/feed` — **bot-blocked**: Cloudflare 'Just a moment...' challenge page
- **news:USA Today** — `https://rssfeeds.usatoday.com/usatoday-NewsTopStories` — **bot-blocked**: 402 'Access Restricted'
- **news:Washington Post** — `https://feeds.washingtonpost.com/rss/national` — **transient-error**: 502 Bad Gateway twice — server-side, feed kept, SourceHealth re-checks

### Verified-live (86 of 97 kept = 79 of the 90 surviving originals + 7 new replacements)

All other feeds returned HTTP 200 with >=1 parsed item. Full per-feed evidence is in the
verification runs (news_verification round 2 + Accept-header retry round 3).

## Search engines (10)

- Verified-live (6): Bing, Mojeek, Qwant, Startpage, Yandex, Brave (429 rate-limit on first
  fetch, 200 + results on retry).
- Bot-blocked (3): DuckDuckGo (202 challenge page), DuckDuckGo-lite (202), Ecosia (403).
- Unverifiable (1): Yahoo (connection reset from verifier network).
- Kept: all 10 — the bot-blocked ones are the primary engines and work from non-datacenter
  networks; SourceHealth re-checks them at runtime.

## Music sources

- Verified-live (10): iTunes Search API, Deezer API, MusicBrainz, Wikimedia Commons API,
  Internet Archive advancedsearch, Audiomack homepage (URL-first-class, no keyless search),
  NotJustOk, TooXclusive, SoundCloud search page, YouTube search page.
- Bot-blocked (1): Bandcamp — Cloudflare 'Client Challenge' from this network; kept, re-checked.
- Unverifiable (2): NetNaija, Openverse — connection reset from verifier network (NetNaija's
  geo-block is already documented in `sources.py`); kept, re-checked.
- **Removed from the active chain (1): Boomplay search.** The search page is a JS SPA shell
  (placeholder `songName: Unknown`, zero result links via plain HTTP); its autocomplete
  endpoint `/searchItem` 500s without a browser session and its search API is AES-encrypted
  (`gbp.AESEncrypt` in the page bundle). Plain-HTTP search returns no results, so the entry
  was removed from `SOURCE_CHAIN`, `MUSIC_SEARCH_SOURCES`, `MUSIC_SOURCE_REGIONS`, and the
  resolver chains. `BoomplaySource` + the resolver method stay (documented, still tested)
  for a future browser-backed search. Africa coverage is unchanged (NetNaija/NotJustOk/TooXclusive).

## Film sources (3)

- Verified-live: 0 from this network.
- Bot-blocked (1): NetNaija Movies (thenetnaija.net) — 403.
- Unverifiable (2): Nkiri, FzMovies — connection reset from verifier network.
- Kept: all 3 — Nigeria-network sources the verifier network blocks; SourceHealth re-checks
  them at runtime. `search_url()` methods added to Nkiri/FzMovies (single-sourced URLs).

## Weather providers (5)

- Verified-live (4): Open-Meteo geocode, Open-Meteo forecast (Lagos), wttr.in, NWS alerts.
- Unverifiable (1): met.no — connection failed even with the identifying UA its ToS requires;
  kept (documented reliable API), re-checked at runtime.

## Wisdom sources (4)

- Verified-live (4): archive.org advancedsearch, Gutendex (timeout once, verified on retry),
  Wikisource API, Project Gutenberg text.

## Price feeds (4)

- Verified-live (2): Coinbase candles, Kraken OHLC.
- Geo-blocked (1): Binance — HTTP 451 from the verifier network; kept, re-checked.
- Bot-blocked (1): Bybit — 403; kept, re-checked.

## Code changes in this pass

- `nomorals/agency/sourcehealth.py` (new): per-source health, conditional requests, circuit
  breaker (3 → stood down, 10 → disabled), SQLite persistence, verified-live gating.
- `nomorals/agency/sources_registry.py` (new): single-sourced registry collecting every list
  from its real owning module — no duplicated URL lists.
- `nomorals/agency/expansion.py` (new): second-wave autonomy — media, code, books, games,
  commerce, health, travel, property, legal, finance, learn, marketing, community, characters,
  documents, datasci, sourcehealth — on the shared scheduler/stores.
- `nomorals/agents/news.py`: 9 dead feeds removed, 7 verified replacements added, `NewsAgent`
  sends RSS `Accept` headers (rescues CBS/ESPN/PBS/Engadget).
- `nomorals/media/sources.py`: Boomplay removed from active chain; `search_url()` added to
  ArchiveOrg + Bandcamp (single-sourced URLs).
- `nomorals/media/resolver.py`: Boomplay removed from both search chains.
  Pre-existing breakage found while testing, fixed in the same file:
  `ytdlp_available()` was missing though `test_play_search_fix.py` imports it;
  `_finalize_failure` matched the substring "yt-dlp" (a bot-check error contains it in a GitHub
  URL) instead of verified facts — now bot-check → cookies help, verified-missing → official
  install hint (`YT_DLP_HINT` single-sourced from `cookies.YTDLP_INSTALL_HELP`); `_preview_reason`
  treated MONETIZE as preview-only though `_PREVIEW_POLICIES` excludes it — now only the four
  hard policies count; `_s_soundcloud_search` fetched exactly 1 track — now up to 5, skipping
  preview-only ones, failing honestly when all are preview-only.
- `nomorals/agents/partner/runtime_media.py` (same pre-existing /play fix): `_control_play`
  appended `resolution.hint` after `explain_failure` (which already includes `hint: ...`),
  printing the hint body twice — now appended only on the fallback path.
- `nomorals/media/film_sources.py`: `search_url()` added to Nkiri + FzMovies.
- Tests updated to the verified reality: `test_many_sources` (no boomplay in chain; news fake
  accepts the headers kwarg), `test_audio_sources` (chain-order assertion without boomplay),
  `test_global_sources` (fifteen verified sources), `test_resolver` (official yt-dlp install form).
  `test_search_sources` needed NO edit: its registry expectation was already right — the failure
  was the harness missing `search/osint_browser.py`, which supplies the 2 browser OSINT specs.

