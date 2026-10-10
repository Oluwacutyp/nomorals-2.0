# TRIGGERS sweep — external mining report

Module: `nomorals/triggers/` (8 files). Mined 2026-10-10 BEFORE any code.
Rule: for every significant class, "how does the best implementation of X do it?"

## Sources consulted

- Home Assistant automations (trigger/condition/action split, modes, blueprints)
- Huginn (agent chains, Liquid templating, digest agents, validate_options)
- APScheduler / Quartz / n8n scheduler (misfire policies, coalesce, jitter, max_instances)
- changedetection.io (URL watching, selectors, JSONPath, text triggers, notification templates)
- Stripe / GitHub / Svix webhook security (HMAC over `ts.body`, 300s timestamp tolerance, idempotency keys)
- CloudEvents 1.0 (event envelope: id/source/type/specversion/time/subject/data + correlation/causation extensions)
- watchdog / inotify (event-driven file watching vs polling)
- Lamudi / Jiji saved-search alert patterns (already baked in; Jiji snooze/digest lessons)

## Per-class findings

### TriggerEngine (engine.py)

- **Home Assistant**: separates *trigger* (when) from *condition* (gate evaluated
  after trigger, before action). Ours has no conditions — a trigger that matches
  ALWAYS fires (modulo cooldown). → ADD `conditions` list on the trigger
  (time_window, rate cap, evidence_match), evaluated in `_fire`.
- **HA modes**: `single | parallel | queued | restart` concurrency control per
  automation. Ours has no execution mode — a slow action can re-fire and stack.
  → ADD `mode` (`parallel` default, `single` skip-while-running, `queued` serialize).
- **n8n scheduler**: misfire policies (`skip | coalesce | catch-up`), misfire grace
  windows. Our schedule triggers silently miss runs while the engine is down.
  → ADD `misfire_policy` (`skip` default keeps today's behavior; `fire_once` fires
  on engine start when the last scheduled run was missed) + `next_run()` inspection
  (CronParser.next_run already exists in-repo — use it).
- **CloudEvents**: `id + source` is the consumer dedup key. Our bus already has
  a depth cap; envelope mapping to CE-1.0 for external forwarding is missing.
  → ADD `to_cloudevent(event)` in sources.py.
- **HA blueprints**: reusable automation templates with inputs. Ours has no templates.
  → ADD built-in trigger templates + `add_from_template()`.

### Trigger models (models.py)

- HA `numeric_state` has above/below with hysteresis; message triggers elsewhere
  support `exclude` (negative patterns) and case-insensitivity. → ADD `exclude` regex
  and `case_insensitive` to message conditions.
- HA trigger names/ids must correspond and be stable. Ours: fine already.
- Validation is already fail-fast (best practice, matches Huginn's
  `validate_options`). Keep.

### Sources (sources.py)

- **changedetection.io**: website change monitoring is THE product here — watch a URL,
  extract via CSS selector / JSONPath / regex, fire on change, fire only when the
  new text matches a trigger regex, ignore noise regexes. We have NO url source —
  file watching only. → ADD `SOURCE_URL` ("url"): stdlib fetch, content extraction,
  hash baseline, change evidence with snippet + diff size.
- **watchdog**: event-driven FS watching. Polling with sha256 is fine and portable;
  keep polling, note inotify as a future.
- `match_message`: add negative/exclude pattern + case-insensitive option (HA parity).
- `evaluate_price`: fine; watchers' condition engine is the right reuse.

### Actions (actions.py)

- **Huginn Liquid / changedetection.io templates**: action params are formatted with
  event data (`{{current_price}}`, `{{watch_url}}`). Ours are STATIC — a price alert
  body can't include the price! Biggest UX gap in the module. → ADD evidence
  templating (`{{price}}`, `{{match.title}}`, dotted paths, safe defaults) applied
  to notify title/body, message text, mission goal.
- **Huginn digest agents**: batch N events into one summary push instead of N pings.
  → ADD `digest` support: a per-trigger digest buffer that flushes on schedule or
  size threshold (`notify`/`message` actions with `digest: true` params accumulate,
  `flush_digests()` sends one combined message). Default off.
- HA `continue_on_error` / choose blocks: out of scope; the fail-closed record is
  already good.

### Store (store.py)

- Webhook idempotency (Stripe pattern): dedupe table for processed webhook event
  IDs with TTL. → ADD `trigger_webhook_events` table + `note_webhook_event` /
  `seen_webhook_event`.
- History TTL per trigger + `stats()` (fires/errors by outcome) for the digest/
  status surface. → ADD `stats(trigger_id)` and keep global TTL.

### Webhook (webhook.py)

- Current auth = shared secret compare only. Stripe/GH/Svix standard:
  HMAC-SHA256 over `{timestamp}.{raw_body}`, 300s tolerance, constant-time compare,
  raw body bytes (not re-serialized), idempotency keys, fast 200 + async processing.
  → ADD `hmac` scheme to webhook conditions:
  `{"secret": ..., "scheme": "github"|"stripe"|"plain", "tolerance_s": 300}`;
  engine verifies signature headers + timestamp; store dedupes by event id.
  Backward compatible: `{"secret": ...}` alone keeps the old behavior.

### Saved search (saved_search.py)

- Jiji pattern: snooze a watch; digest mode instead of instant pushes. → ADD
  `snooze(search_id, hours)` and `digest_at` ("HH:MM") batching: matches queue in
  `pending_digest`, `deliver_digests()` pushes one combined message per search at
  the configured hour.
- kept: Lamudi stale-demand hygiene (expiry), dedup (photo-hash + normalized address).

### Display / style

- changedetection.io MCP + HA UIs show watch status with next-check countdown and
  diff preview. CLI output today is flat text. → ADD `display.py`: outcome glyphs
  (✅ fired / ⚪ no_match / ⏸️ skipped / ❌ error), rich one-liner per trigger
  (status, source→action, next run, fires, last outcome), detail box, history row
  formatter, plain/color themes.

## What stays as-is

- Polling architecture for file/price (portable; matches watchers.FileKind semantics).
- Fail-fast validation at creation (Huginn-parity best practice).
- Scheduler reuse for schedule sources (no parallel scheduler — right call).
- Per-trigger isolation + history for every outcome (no silent drops — right call).
- Ledger + telemetry events (keep, extend lightly).

## Weak spots honestly noted

- No real inotify/watchdog backend (poll-only). Acceptable for portability; noted.
- `mode: restart` not implemented (needs action cancellation machinery — out of scope).
- URL source uses stdlib urllib (no Playwright/JS rendering) — JS-heavy pages won't
  render; documented in the validator message.
