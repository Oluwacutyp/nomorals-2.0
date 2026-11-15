# Devon web — Vercel frontend

Devon's web interface, served from Vercel. Plain HTML/CSS/JS, no build step,
no framework, no dependency. It talks to Devon's HTTP API, which stays on
her own server — this page is the front, Devon is the back.

## What it is

Three views, all real, all against Devon's actual API
(`nomorals/api/server.py`):

- **Chat** — sends messages to `POST /chat` (her model router) and shows the
  reply, with the model, provider, latency, and any degradation flag the
  router reports. The conversation is kept in the page so follow-ups have
  context. Temperature and max tokens are adjustable.
- **Status** — `GET /health` (version, schema, uptime, providers, runtime
  beacon), `GET /ready` (database), `GET /events` (recent activity,
  auto-refreshes every 15s), `GET /tools` (what the saved token may call),
  plus links to her public pages (`/devon`, `/rules`) on the API address.
- **Pairing** — `GET /pair/state` and `POST /pair/action`: every account as a
  card with its status, live toggle verdicts, WhatsApp pairing code,
  Telegram login flow (code + 2FA password), the nine personal toggles as
  switches, lifecycle buttons (start / stop / restart / remove / use), and a
  form to pair a new WhatsApp or Telegram account. Every action returns the
  fresh state, so the view re-renders from the truth in one round trip.

## What it is not

- It does **not** run Devon. She stays on her own server (EC2) with her
  database, scheduler, and bridges. Vercel is serverless and cannot host a
  persistent agent.
- The chat talks to her **model router** (`POST /chat`), not the full chat
  runtime. Slash commands, games, and the rest of her behaviour live in
  Telegram and WhatsApp. This page is for conversation, status, and pairing.
- It does not use the live event stream (`GET /stream`, SSE). A browser
  `EventSource` cannot send the Bearer token the stream requires, so the
  status view polls `GET /events` instead.

## Deploy on Vercel

1. In Vercel, import this repo (`Oluwacutyp/nomorals-2.0`), branch `main`.
2. Leave **Root Directory** at the repo root (`./`). The `vercel.json` at the
   repo root points Vercel at this folder (`outputDirectory: web/vercel`) and
   sets the framework to none — no build step, the files are served as-is.
3. Deploy. The site is static; there is nothing to build.
4. Open the site, open **Settings**, and enter:
   - **Devon API address** — her API on the server she runs on, with the
     port. Example: `https://203.0.113.10:8731`. No trailing slash. The port
     is `api.port` in her settings (8731 by default).
   - **API token** — her API token (`NM_API_TOKEN`). The owner token gives
     full control of Devon.
   - **How this page reaches her API** — Direct or Vercel proxy (below).
5. Press **Test connection**. It calls `GET /live` (no token needed) and then
   `GET /health` (needs the token) and reports what it finds.

The address and token live in **your browser's localStorage only**. They are
never written into the repo, never sent anywhere except Devon's API, and
never leave your machine. Anyone holding the token runs Devon, so do not
share the browser profile you save it in. If the site itself must not be
public, add Vercel's password protection to the project.

## How the page reaches her API — two modes

**Direct (default).** The browser calls her API address straight. This is the
clean architecture. It needs her API to allow this site's origin — see
**CORS** below. Until CORS is set, the browser blocks the calls and the page
says so (it names CORS as the likely cause).

**Vercel proxy.** The browser calls this same site under `/api/…` and Vercel
forwards it to her API, server to server. The browser never talks to her API
directly, so no CORS is needed and nothing on her server has to change. To
use it, add this rewrite to the `vercel.json` at the repo root and redeploy:

```json
{
  "rewrites": [
    { "source": "/api/:match*", "destination": "https://YOUR-EC2-API-HOST:8731/:match*" }
  ]
}
```

Replace `YOUR-EC2-API-HOST:8731` with her real API host and port. Then pick
**Vercel proxy** in Settings. In proxy mode the API address field is ignored
(the rewrite decides where the calls go); the token still comes from Settings.

## CORS — what her API needs (Direct mode)

Direct mode needs her API to send `Access-Control-Allow-Origin` for this
site's address. Her API server supports it — but read this carefully,
because the setting is **not wired up yet**:

- The server has the switch: `cors_origins` on `APIServer`
  (`nomorals/api/server.py:396`, echoed at `:1194-1225`, preflight handled
  at `:1596-1617`). When set, it echoes a matching origin and answers
  `OPTIONS` preflights with `Access-Control-Allow-Methods` and
  `Access-Control-Allow-Headers: Authorization, Content-Type, X-Request-ID`.
- The config field exists: `api.cors_origins` (`nomorals/core/config.py:709`,
  default `"*"`).
- **But nothing passes it through.** `serve()` (`nomorals/api/server.py:1642`)
  does not pass `cors_origins` to `APIServer`, `nm serve`
  (`nomorals/cmdline/commands/serve.py`) does not pass it either, and there
  is no `NM_API_CORS_*` environment variable. On a stock `nm serve` run,
  CORS is **off** and the config field is dead. Setting `api.cors_origins`
  alone changes nothing today.

The fix is one small wiring change on her server (a follow-up, not part of
this frontend — this lane may not modify the API server): pass the config
through in `serve()`, e.g.

```python
api = APIServer(
    context,
    token=token or context.settings.api.token,
    principals=principals,
    max_body_bytes=max_body_bytes,
    cors_origins=[o.strip() for o in
                  (context.settings.api.cors_origins or "").split(",")
                  if o.strip()],
)
```

Then set `api.cors_origins` to this site's address — e.g.
`https://devon-web.vercel.app` (strict, recommended) — or leave the default
`"*"` (any origin may call; the Bearer token is still required for every
route except `/live`, so an unknown site still cannot act without it).
Restart her API after the change.

Until that wiring lands, **use Vercel proxy mode** — it works with no change
on her server.

## Her server must be reachable

Whichever mode you use, her API has to be reachable from the internet: the
API port (`api.port`, 8731 by default) must be open in the server's firewall
/ security group, and the address you enter must be one the browser (Direct)
or Vercel's edge (proxy) can reach. If the API only listens on loopback
(`127.0.0.1`), neither mode can reach it — it must listen on `0.0.0.0` (her
settings default) with the port open.

## Files

- `index.html` — the page: Chat / Status / Pairing views + Settings.
- `app.js` — the API client and all view logic. No dependency.
- `styles.css` — vanilla CSS, no build step.
- `README.md` — this file.
- `../vercel.json` (repo root) — `outputDirectory: web/vercel`,
  `framework: null`, `cleanUrls: true`. The proxy rewrite is **not** in it
  by default; add it from the snippet above if you use proxy mode.

## Endpoints used (all real — read from `nomorals/api/server.py`)

| Call | Route | Auth | What for |
|---|---|---|---|
| `GET /live` | server.py:568 | none (public) | Is anything answering? |
| `GET /health` | server.py:622 | token | Version, schema, uptime, providers, runtime beacon |
| `GET /ready` | server.py:607 | token | Database answers? |
| `POST /chat` | server.py:747 | `MODEL_CALL` | Chat with the model router |
| `GET /events` | server.py:823 | `DB_READ` | Recent activity (polled) |
| `GET /tools` | server.py:685 | token (filtered to grant) | Tools the token may call |
| `GET /pair/state` | server.py:1145 | `SYS_CONFIG` (owner) | Pairing pool state |
| `POST /pair/action` | server.py:1157 | `SYS_CONFIG` (owner) | Pairing actions |

Pairing actions (`nomorals/social/pairing_web.py:238`): `pair_whatsapp`
`{number, label}`, `pair_telegram` `{number, label}`, `tgcode` `{id, code}`,
`tgpass` `{id, password}`, `start` / `stop` / `restart` / `remove` / `use`
`{id}`, `toggle` `{id, name, on}`. Every action returns the fresh state.

Chat response shape (`nomorals/llm/base.py:440` `LLMResponse.to_dict`):
`{text, model, provider, usage, finish_reason, latency_ms, error, degraded,
failed_providers, fallback_note, failure_class, route_trace, cost_usd,
adaptation_note, deferred}`. A non-empty `error` means the model call failed
even though the HTTP status was 200 — the page shows it, and does not keep a
failed turn in the conversation.

## Honest limits

- The page cannot reach her API at all until either CORS is wired (Direct)
  or the proxy rewrite is added (proxy). Both are documented above; neither
  is faked — if the connection is not set up, the page says so and names the
  cause instead of showing a spinner forever.
- The chat is the model router, not the full runtime. It will not run slash
  commands or games, and it does not see Telegram/WhatsApp history.
- The status view polls; it is not a live stream (see "What it is not").
- The token is stored in the browser. That is the standard approach for a
  personal dashboard, but it means anyone with access to that browser
  profile holds the owner token. Use a private profile and do not share it.
