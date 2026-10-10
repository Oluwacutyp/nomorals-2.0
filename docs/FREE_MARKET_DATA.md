# Free market-data endpoints (Sentinel finance brain)

Devon's finance layer pulls market data **keyless-first**: everything works
with no signup and no API key. Keyed free tiers are optional upgrades via
environment variables — never required.

## Default chains (``source="auto"``)

| Market | Chain (first success wins) | What you get |
|---|---|---|
| crypto | Binance → Kraken → Coinbase → CoinGecko | OHLCV klines, 1m–1M, up to 3000 bars (paginated) |
| stocks | Yahoo → Stooq → *(keyed upgrades if env set)* | intraday+daily OHLCV via Yahoo v8 chart API (keyless, no package); Stooq daily as fallback |
| forex | Frankfurter → open.er-api → ECB → Yahoo → Stooq → *(keyed)* | ECB-blend daily fixings; open.er-api live rates; ECB direct XML; Yahoo intraday via `EURUSD=X` |
| commodities | gold-api → Yahoo futures → Frankfurter | real-time XAU/XAG/XPT/XPD spot; GC=F/SI=F/PL=F/PA=F OHLC; daily fixings |

## Source reliability tiers

Infrastructure first — these are the least likely to disappear:

| Tier | Sources | Why |
|---|---|---|
| 1 — Infrastructure | Binance, Kraken, Coinbase (exchanges); ECB, Frankfurter (central banks) | Exchanges and central banks are the data originators |
| 2 — Established aggregators | Yahoo, CoinGecko, Stooq | decade+ track records, widely depended upon |
| 3 — Community utilities | gold-api.com, open.er-api.com | verified working keyless 2026-10-10, smaller operations |

Every source has automatic health tracking: after 3 consecutive failures
it's skipped for 5 minutes (circuit breaker), routing around dead sources
without manual intervention. See `source_health()` in `market_data.py`.

## Keyless sources

| Source | Endpoint | Coverage | Rate limit | Notes |
|---|---|---|---|---|
| Binance | `api.binance.com/api/v3/klines` | crypto OHLCV, 14 intervals | ~1200 weight/min, no key | default for crypto; geo-blocked in the US → auto-falls-back |
| Kraken | `api.kraken.com/0/public/OHLC` | crypto OHLC, 8 intervals | public tier, no key | |
| Coinbase | `api.exchange.coinbase.com/.../candles` | crypto, 6 granularities | 10 req/s public, no key | |
| CoinGecko | `api.coingecko.com/api/v3/coins/{id}/ohlc` | crypto OHLC, **no volume** | ~5–15 calls/min free | granularity follows range (30m/4h/4d) |
| Stooq | `stooq.com/q/d/l/?s=aapl.us&i=d` | stocks daily OHLCV; forex `eurusd` daily | generous, no key | bot-walled from datacenter IPs — kept as fallback |
| Yahoo | `query1.finance.yahoo.com/v8/finance/chart/{sym}` | stocks/forex/crypto intraday+daily | unofficial, no key (~2000 req/hr tolerated) | default for stocks; forex via `EURUSD=X` |
| Frankfurter | `api.frankfurter.dev/v2/rates` (time series), `/v2/rate/{base}/{quote}` | fiat FX + XAU/XAG daily (central-bank sources) | no quotas, no key | one fixing/day: O=H=L=C, volume 0; covers XAU, XAG, NGN |
| gold-api.com | `api.gold-api.com/price/{XAU,XAG,XPT,XPD}` | real-time metals spot | no key, CORS-enabled | verified 2026-10-10; /price/ keyless, /history/ needs key |
| open.er-api.com | `open.er-api.com/v6/latest/{base}` | fiat FX live rates | no key | exchangerate-api.com free tier; forex redundancy |
| ECB direct | `ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml` | EUR reference rates | no key, no limit | the actual source; updated ~16:00 CET working days |
| Yahoo metals | `query1.finance.yahoo.com/v8/finance/chart/GC=F` | GC=F/SI=F/PL=F/PA=F futures OHLC | unofficial, no key | XAUUSD=X is delisted — futures are the working path |

Quotes use the same hosts: Binance 24h ticker for crypto (real 24h change %),
gold-api.com for metals spot (real-time), Frankfurter → open.er-api → ECB
for FX, Yahoo v8 chart for stocks.

## Optional keyed upgrades (env vars)

Set the variable to activate; the source joins the front of the auto chain
for its markets. Nothing breaks when they are absent.

| Env var | Source | Free tier | Best for |
|---|---|---|---|
| `TWELVEDATA_API_KEY` | TwelveData | 8 req/min, 800 credits/day | stocks/forex intraday bars |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage | 25 req/day | stocks/forex daily (also crypto daily) |
| `FINNHUB_API_KEY` | Finnhub | 60 calls/min | crypto/stocks/forex candles |
| `NEWSAPI_KEY` | NewsAPI.org | 100 req/day | news-sentiment layer (future) |

## What Sentinel still needs keys for (unchanged)

These are Sentinel.py's own keyed integrations, untouched by the rewire:

- **MetaTrader 5** (`MT5_LOGIN`/`MT5_PASSWORD`/`MT5_SERVER`) — the
  `sentinel_bot.py` monolith's broker feed + execution. Only needed if you
  run that standalone bot.
- **Exchange API keys** — only for *placing live orders* via
  `sentinel/live/trader.py:CCXTBroker`. They live in Devon's credential
  vault, never in env/config. Public market data never needs them.
- **Gemini** (`GEMINI_API_KEY`) — optional narrative polish in
  `sentinel_bot.py`; off by default.

## Pinning a source

```python
from nomorals.integrations import sentinel_bridge as bridge
df = bridge.load_data("BTC/USDT", "crypto", "1h", 500, source="binance")
df = bridge.load_data("AAPL", "stocks", "1d", 500, source="stooq")
df = bridge.load_data("EURUSD", "forex", "1d", 500, source="frankfurter")
# legacy package paths still available:
df = bridge.load_data("BTC/USDT", "crypto", source="ccxt")      # needs ccxt
df = bridge.load_data("AAPL", "stocks", source="yfinance")      # needs yfinance
```
