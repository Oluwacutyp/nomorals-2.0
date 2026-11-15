# Devon Interface — WordPress plugin

Devon's web interface, on a WordPress site.

Devon is a persistent personal AI agent. She runs on her own host
(a VM) with her database, her scheduler, her WhatsApp bridge, and
her Python API server. **WordPress cannot host Devon herself** —
and this plugin does not try. It renders her web interface on a
WordPress site by calling her API, which stays where she lives.
WordPress holds the front, Devon holds the thinking.

## What it does

Three views, all real, all against Devon's actual API:

- **Chat** — a conversation with Devon's brain (her model router,
  `POST /chat`). The reply carries its model, provider, latency,
  and any degradation note (when a fallback provider served it).
  The history lives in the browser and is sent whole each turn;
  the API is stateless. The full partner turn — tools, memory,
  platform context — lives in Telegram and WhatsApp; this chat is
  her brain answering, and the interface says so.
- **Status** — her live status: `GET /live` (is she answering?),
  `GET /health` (version, schema, process id, uptime, provider
  list, runtime state — alive / stopped / stale / not running —
  beacon age, last error), and the pairing counts (registered /
  connected / dropped). Each leg is reported on its own, so a
  bridge being down does not hide the health.
- **Pairing** — the pairing pool, on the same contract Devon's
  own pairing page drives (`GET /pair/state` + `POST /pair/action`,
  see `web/pairing.php`): pair WhatsApp and Telegram accounts,
  see live pairing codes and pending Telegram login flows, answer
  login codes and 2FA passwords, flip the nine personal toggles,
  start / stop / restart / remove accounts, set the default.
  Admin-only. Pairing is approved on the phone, always — a web
  page can show a code, it can never approve one.

## Install

1. Copy the `devon-interface` folder into your WordPress
   `wp-content/plugins/` directory (or zip it and upload it via
   Plugins → Add New → Upload Plugin).
2. Activate it (Plugins → Installed Plugins → Devon Interface →
   Activate).
3. Go to Settings → Devon Interface and set:
   - **API base URL** — where Devon serves her API, e.g.
     `http://your-ec2-address:8787`. The default port is 8787.
     WordPress must be able to reach that address over the network.
   - **API token** — the owner API token (`NM_API_TOKEN` on
     Devon's host). The one with full power. It is stored in the
     WordPress database, never shown back, and never sent to the
     browser.
4. Click **Test connection**. It calls `GET /live` (no auth) and
   `GET /health` (needs the token) and reports exactly what Devon
   answered, so a wrong address or a wrong token is visible here
   before any shortcode is embedded.

## Embed

Put a shortcode in any page or post:

- `[devon_chat]` — chat with Devon's brain.
- `[devon_status]` — her live status.
- `[devon_pairing]` — paired accounts (admins only; anyone else
  sees "Pairing is owner-only.").
- `[devon_interface]` — all three, on one page.

Assets (CSS + JS) are enqueued only on pages that carry one of
the shortcodes.

## Architecture — why CORS is not needed

Every call to Devon's API is made **server-side, from PHP**,
with the owner token in the `Authorization` header. The browser
talks only to WordPress (`admin-ajax.php`, with a nonce), and
WordPress talks to Devon. The token never reaches the browser.

Because the calls are server-to-server, **CORS is not required**:
the browser never calls Devon's API directly, so no `Origin`
check ever runs. Devon's API keeps its CORS off (the default).
There is nothing to configure on EC2 for this plugin to work.

For the record: Devon's API does have a CORS switch
(`cors_origins` on the API server), and a `cors_origins: "*"`
setting exists in her config (`nomorals/core/config.py`), but
that setting is **not wired into the shipped `nm serve` path** —
nothing passes it to the API server, so the API serves no CORS
headers today. If direct browser-to-Devon calls are ever wanted,
that is a separate, small change on Devon's side (wire the
setting into `serve()`). This plugin deliberately does not need
it, and does not ask for it: the server-side proxy is the safer
shape, because it keeps the owner token off the browser.

## Security

- **Pairing is admin-only.** The pairing contract is owner-only
  on Devon's side (`SYS_CONFIG` capability), and this plugin
  enforces the same: the pairing shortcode renders only for
  admins, and the pairing AJAX handlers require
  `manage_options`. Non-admins never see pool state.
- **Chat and status** are available wherever the owner embeds
  the shortcode — the owner chooses the page. Both use the owner
  token server-side, so anyone who can use the chat shortcode is
  chatting with Devon's brain on the owner's token. Do not put
  `[devon_chat]` on a public page unless you mean for visitors
  to chat with her.
- **The plugin owns nothing.** Every render is Devon's API read
  fresh. If her API is down, refuses the token (403), or answers
  with something that is not JSON, the plugin says so in plain
  words — it never invents a status, a reply, or a pairing code.
- All AJAX calls carry a WordPress nonce. All settings are
  sanitized (`esc_url_raw` for the URL, `sanitize_text_field`
  for the token). All output is escaped.

## API endpoints used

All read from `nomorals/api/server.py`:

| Endpoint | Used for |
|---|---|
| `GET /live` | liveness probe; settings-page connection test |
| `GET /ready` | (available; not used by the plugin) |
| `GET /health` | status view + connection test |
| `GET /docs` | (available; not used by the plugin) |
| `POST /chat` | chat view (model router) |
| `GET /pair/state` | pairing view + status counts |
| `POST /pair/action` | pairing actions |
| `GET /devon`, `GET /rules` | Devon's public pages — served by her API directly, not by this plugin |
| `POST /site/publish` | Devon's own WordPress publishing (she pushes her page + rules pages to WordPress) — the reverse direction; this plugin complements it, it does not duplicate it |

The pairing contract (`GET /pair/state`, `POST /pair/action`)
is implemented in `nomorals/social/pairing_web.py`
(`pairing_state`, `pairing_action`); the nine personal toggles
are `PERSONAL_TOGGLES` in `nomorals/social/pairing.py`
(`status_saving`, `view_once`, `antidelete`, `always_online`,
`status_viewing`, `status_react`, `anticall`, `antitag`,
`antiedit`). The chat response shape is `LLMResponse.to_dict()`
in `nomorals/llm/base.py` (`text`, `model`, `provider`, `usage`,
`finish_reason`, `latency_ms`, `error`, `degraded`,
`failed_providers`, `fallback_note`, `failure_class`,
`route_trace`, `cost_usd`, `adaptation_note`, `deferred`).

## Honest limits

- **Chat is her brain, not the full partner.** `POST /chat`
  goes to the model router. It does not run the partner turn
  (no tools, no memory writes, no platform context). There is
  no API endpoint that routes a message through the full partner
  runtime — the partner lives in Telegram and WhatsApp. The
  interface labels the chat honestly and the README says so.
- **No live event stream.** The API has `GET /stream` (SSE live
  Timeline events), but a server-side PHP handler cannot hold
  an SSE connection open for the browser the way a direct
  connection could. The plugin polls instead (status every 30s,
  pairing every 5s, and only re-renders when the visible truth
  changes). That is the right shape for a WordPress plugin.
- **Not verified against a live WordPress install.** The PHP
  parses clean (`php -l`), the shortcodes, settings, and AJAX
  handlers are wired by reading the code, and every endpoint
  and field was confirmed in Devon's source. But there is no
  WordPress on this box, so activation, the settings page, and
  the shortcode render have not been run against a real
  WordPress. The first install on a real site is the proof.
  The same holds for the EC2 side: the plugin needs Devon's API
  reachable from the WordPress host at the configured address.
