---
id: trading-read
title: Trading read — whole market, honest provenance
summary: How a market read is taken: every timeframe independently, stale data named, never filled from another frame.
topics: trading, goldenswing, market, xauusd
triggers: trade read, market read, price check, xauusd, gold price
layer: ta
version: 1
---

# Trading read

- A read is whole-market: D1 through M5 plus the live price, each
  timeframe's state computed independently, with cross-timeframe
  relationships stated — never one frame dressed as another.
- Provenance is honest: a stale or missing timeframe is marked stale or
  missing. Data is never filled from a different timeframe to make the
  read look complete.
- Every read journals. Judgment consumes reads as evidence; forecasts
  are structured per horizon (direction, point price, interval,
  excursions, disagreement, scenario and invalidation) — percentile
  ranks alone are not a forecast.
- The pipeline never places orders. Placements happen only through the
  execution pass the owner pokes; judgment's order authority is a
  separate, explicit promotion decision.
