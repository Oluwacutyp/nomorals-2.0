# The Descriptor Rule

> **A new capability of kind X is a new descriptor, not a new code path.**

## The rule

When a feature varies by *kind* — platform, provider, model, channel,
locale, game — the variation lives in **data** (a descriptor row), and the
code that consumes it is written **once**, against the descriptor's fields.
Adding a new kind means adding a row. It never means adding a branch,
a special case, a new `if platform == "x"`, or a copy of a function with
different constants.

Concretely:

- **DO:** `limits = for_platform(chat.platform)` then `chunk_message(text, max_text=limits.max_text)`.
- **DO NOT:** `if platform == "telegram": limit = 4096 elif platform == "discord": limit = 2000 …`.
- **DO:** `register_limits(PlatformLimits(platform="signal", …))` to add a platform.
- **DO NOT:** add a `signal.py` adapter + a gateway branch + a chunker branch
  just to express "Signal caps text at N".

## Why

The codebase already paid for the opposite shape twice: media handling and
transmission each looked like they needed new subsystems, and discovery
showed both were ~70% built behind scattered constants (`4096` in
`telegram.py`, `4000` in `whatsapp.py`, `2000` implied in `discord.py`,
`160` in `sms.py`, `3000`/`4000`/`1500` in a second table in
`platforms.py`). Every hardcoded constant was a place a new platform
had to be re-derived, re-tested, and re-broken. One table, one consumer,
one test: a new platform is one row.

## What counts as a descriptor

A descriptor is a frozen, declarative record of *what a kind can do and
what it costs* — limits, capabilities, budgets, cost models. It answers
questions; it never performs I/O, never branches on identity, never
imports an adapter. Examples:

| Kind | Descriptor | Consumer |
|---|---|---|
| Chat platform | `PlatformLimits` (`social/chat/platform_limits.py`) | gateway chunking, media routing, retry budgets |
| (future) Model | model card: context, cost, capabilities | brain routing |
| (future) Download source | source card: auth, rate, formats | downloader |

## First adopter: `PlatformLimits`

`nomorals/social/chat/platform_limits.py` is the first descriptor table
under this rule. It owns, per platform: `max_text`, `max_caption`,
`max_file_mb`, `supports_buttons`, `supports_html`, `supports_media_edit`,
`rate_per_hour`, `cost_model`, plus `gateway_chunks` (False when the
adapter owns segmentation, e.g. SMS's encoding-aware splitter).

Every consumer in the transmission path reads from it:

- `social/chat/chunking.py` — chunking + media-as-document decisions.
- `social/chat/gateway.py` — send/send_file chunking, button/format gating.
- `social/chat/telegram.py`, `whatsapp.py`, `sms.py`, `progress_throttle.py` —
  their former hardcoded caps now derive from the table.
- `social/chat/platforms.py` — the old `PLATFORM_LIMITS` rendering-budget
  dict is now derived from the same table (same values for the hard caps;
  callers that want a conservative budget pass their own limit).

## What is NOT a descriptor (and stays in code)

- **Protocol mechanics.** GSM-7/UCS-2 segment math in `sms.py` is the GSM
  spec, not platform policy — it stays in the SMS adapter. Telethon/Bot API
  call shapes stay in the Telegram adapter. A descriptor says *what*; the
  adapter still owns *how*.
- **Per-message choices.** Whether *this* send is durable, what its
  `durable_kind` is (report/media/digest/chat), and its idempotency key
  are per-send decisions, passed as arguments — not descriptor fields.
- **Secrets and credentials.** Never in a descriptor. Adapters take them
  in the constructor from settings/env.

## Enforcement

1. **Conformance test** (`tests/test_adapter_conformance.py`) — every
   `ChatAdapter` subclass must accept the full base `send()` signature.
   A new adapter that narrows the contract fails loudly instead of
   breaking at runtime.
2. **Limits tests** (`tests/test_platform_limits.py`) — every known
   platform has a row; aliases resolve; unknown platforms get the
   conservative default, never a crash, never a guessed branch.
3. **Review rule.** A diff that adds `if <kind> == …` (or a new constant
   duplicating a descriptor field) to a consumer is a review failure:
   the constant belongs in the descriptor table, the branch belongs in
   the one consumer that already handles that field.

## Adding a new platform (checklist)

1. Add one `PlatformLimits(...)` row in `platform_limits.py`
   (or `register_limits(...)` at startup for out-of-tree platforms).
2. Add the adapter (it implements the existing `ChatAdapter` contract —
   no gateway change).
3. Add one row to the conformance/limits tests.
4. Done. If step 2 needed a gateway or chunker change, the descriptor is
   missing a field — add the field, don't add the branch.
