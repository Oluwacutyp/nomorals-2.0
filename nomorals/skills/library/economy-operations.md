---
id: economy-operations
title: Economy operations — the vault is not the purse
summary: How coin/economy work stays safe: plan-first settlement, durable claims, consequences bound to the right store.
topics: economy, coins, settlement, wallet, heist
triggers: economy, coins, balance, settle, payout, wager
layer: commerce
version: 1
---

# Economy operations

- Settlement is plan-first and atomic: the settlement row commits
  before any credit moves, in one transaction. A failure mid-settle
  pays nothing; the recovery sweep settles exactly once; a
  crash-after-commit replays receipts and moves zero coins.
- Consequences bind the store that matters: a heist blacklist binds the
  VAULT, not the purse — sanctioning the wrong store is how the
  punished party keeps playing.
- Money movement is an external effect: claim it in the idempotency
  store first (durable-before-effect), so a retry replays the receipt
  instead of paying twice.
- Balances are always read from the store at answer time — never quoted
  from memory, never rounded into a nicer number.
