# STREAM sweep — external mining

Module: `nomorals/stream/` (`server.py`, `errors.py`, `__init__.py`) — a
stdlib-only SSE server streaming Timeline events. Mined 2026-10-10 before
any code was written. All sources below are real and were read during this
sweep; no invented techniques.

## Sources

### 1. sse-starlette — `sysid/sse-starlette` (and `ancieg` fork)
- https://github.com/sysid/sse-starlette (README: ping, send_timeout, fan-out proxies, error handling)
- https://raw.githubusercontent.com/sysid/sse-starlette/main/sse_starlette/sse.py (`EventSourceResponse`)
- https://raw.githubusercontent.com/sysid/sse-starlette/main/sse_starlette/event.py (`ServerSentEvent`)

Mined behaviors:
- **Framing helper**: `ServerSentEvent(data, event, id, retry, comment, sep)`
  splits comments and multi-line data on line boundaries (each line gets its
  own `: `/`data: ` prefix), strips newlines from `id:`/`event:` values,
  requires `retry` to be an int, terminates the event with a blank line.
  Default separator `\r\n`, validated to one of `\r\n`/`\r`/`\n`.
- **Ping**: default ping every 15 s; ping is customizable and can be swapped
  for a *comment* ping (`: ...`) so it is invisible to `EventSource` clients.
- **Send timeout**: `send_timeout` terminates a hanging send — a client that
  holds the connection open but stops reading must not wedge the server
  thread (issue #89 in that repo).
- **Headers**: `Cache-Control: no-store` by default but overridable (fan-out
  proxies rely on cacheable responses — Fastly's SSE guide is cited),
  `Connection: keep-alive`, `X-Accel-Buffering: no` always.
- **Compression is refused**: `enable_compression` raises — gzipping an event
  stream re-batches events.
- **Graceful drain**: cooperative shutdown via a shutdown event + grace
  period so handlers can send farewell events instead of being cancelled
  mid-write; a per-thread shutdown watcher broadcasts to all live streams.
- **Disconnect callback**: `client_close_handler_callable` fires on client
  disconnect for cleanup.

### 2. Mercure protocol — `dunglas/mercure`
- https://github.com/dunglas/mercure/releases (protocol notes)
- https://www.pitsolutions.com/blog/beyond-websocket-real-time-applications-with-mercure-and-sse (practical guide)

Mined behaviors:
- **Hub pattern**: applications publish to a dedicated hub; the hub — not
  the app — holds the long-lived connections and fans updates out. This is
  the fix for our per-subscriber polling loop (each subscriber currently
  opens its own Timeline/SQLite connection every second).
- **Topics**: updates are published on topics; subscribers attach to a topic
  or a topic pattern (Mercure uses URI templates; we use the Timeline's
  existing fnmatch-style globs).
- **Catch-up on reconnect**: SSE `Last-Event-ID` + retained messages =
  automatic catch-up after a drop. Mercure keeps message history for this.
- **Subscription events** carry the SSE `event:` field so clients can route
  with `addEventListener`; control events are namespaced away from data.

### 3. Production SSE handler playbook — lox-solutions/alvyn
- https://github.com/lox-solutions/alvyn/blob/HEAD/website/content/docs/playbook-sse-and-consumer-scaling.mdx

Mined behaviors (Express handler, mapped to stdlib equivalents):
- Headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache,
  no-transform`, `Connection: keep-alive`, `X-Accel-Buffering: no`; flush
  headers immediately (`flushHeaders` → our `end_headers()` + first write).
- **Resume token from two places**: `Last-Event-ID` header *or* a query-param
  fallback; subscription starts from the bookmark with an *exclusive* lower
  bound.
- **AbortController on request close** → release resources on disconnect
  (our equivalent: detect disconnect, unregister subscriber, close queue).
- **15 s heartbeat comment** (`: keepalive`) so intermediate load balancers
  don't drop idle connections.
- W3C framing: `id:`, `event:`, `data: <json>` per event.

### 4. Reverse-proxy pitfalls (nginx/HAProxy/Caddy/Traefik)
- https://dev.to/ji_ai/nginx-proxybuffering-broke-my-llm-sse-stream-04s-became-34s-1fhb
- https://instatunnel.my/blog/fixing-sse-buffer-bloat-in-local-tunnels-guaranteeing-zero-latency-streaming-for-local-llms
- https://dev.to/remdore/nginx-streams-your-tokens-fine-haproxy-holds-them-for-206ms-10p2
- https://dev.to/libme/websocket-closes-every-60-seconds-with-code-1006-finding-the-proxy-idle-timeout-and-fixing-it-with-278e
- https://loadforge.com/guides/api-protocols/load-testing-sesrver-sent-events-sse (load-testing guide)

Mined behaviors:
- `proxy_buffering off` (or `X-Accel-Buffering: no`, nginx-only) for the
  stream location; buffering elsewhere stays on.
- **Never gzip `text/event-stream`** — compression re-batches events.
- `proxy_read_timeout` (default 60 s in nginx) kills idle streams; the fix
  is a server heartbeat well under the timeout (15–25 s is the consensus),
  *then* raising the proxy timeout.
- TCP keepalive does NOT reset L7 idle timers — application-level
  heartbeats (SSE comments) are required.
- `X-Accel-Buffering` is an nginx convention; HAProxy ignores it — don't
  rely on the header alone.
- SSE is constrained by concurrent open sockets (thread-per-connection
  hurts); heartbeat every 15–30 s; load-test reconnect storms (a mass
  reconnect after an outage is the real spike).

### 5. SSE protocol mechanics (WHATWG, via secondary sources)
- https://dev.to/semitexa/server-sent-events-explained-how-sse-works-and-when-to-use-it-87j
- https://github.com/libreguild/zorvik/blob/HEAD/crates/academy/course/10-realtime/03-server-sent-events.mdx (SSE course)
- https://dev.to/jeff_pdc/how-to-test-server-sent-events-endpoints-curl-postman-and-spec-driven-sse-testing-238l

Mined behaviors:
- Event = `event:`/`data:`/`id:`/`retry:` lines terminated by a blank line;
  lines starting with `:` are comments (ignored by clients).
- `id:` sets the client's last-event-ID; the browser re-sends it as
  `Last-Event-ID` on auto-reconnect. `retry:` (ms, digits only) controls
  the reconnect delay.
- "Reconnecting is not the same as recovering": the server must retain
  events and implement the replay policy itself.
- Test with `curl -N` (no-buffer); resume test = reconnect with
  `Last-Event-ID` and assert no dupes/skips.

## Gold → implementation map

| # | Mined gold | Where it lands |
|---|-----------|----------------|
| 1 | `ServerSentEvent` framing (comment/data line splitting, id/event sanitizing, int retry, sep validation) | new `nomorals/stream/sse.py` — own implementation of the spec rules |
| 2 | Hub pattern (Mercure): one poller, fan-out to subscribers | new `nomorals/stream/hub.py` — `EventHub` |
| 3 | Per-subscriber bounded queues + backpressure accounting | `EventHub.subscribe()`; drop-oldest + `dropped` counters surfaced via `stream-warning` event and `/health` |
| 4 | Retained ring buffer + exclusive-lower-bound replay | `EventHub` ring (`seq` per event); `Last-Event-ID` → replay `seq > id`; `?since=` → replay `ts >=` |
| 5 | `retry:` at stream start | `emit_sse` sends `retry: <ms>` first |
| 6 | `Last-Event-ID` header with query-param fallback | `emit_sse` reads header, falls back to `?since=`/`?lastEventId=` |
| 7 | Comment heartbeat 15 s (configurable) | kept, now configurable per server; `: ping` stays client-invisible |
| 8 | Send timeout for hanging clients | socket send timeout in `emit_sse`; `socket.timeout` treated as disconnect |
| 9 | Graceful drain on shutdown | `hub.stop()` closes subscriber queues; `emit_sse` sends `event: shutdown` + `retry` then exits; `StreamServer.stop()` ordering |
| 10 | CORS for dashboard origins + OPTIONS preflight | `StreamServer(cors=...)`, `do_OPTIONS` |
| 11 | TCP_NODELAY / SO_KEEPALIVE on accepted sockets | `get_request` override in server class |
| 12 | Rich health: uptime, subscribers, emitted/dropped | `/health` enriched; `StreamServer.stats()` |
| 13 | `Cache-Control: no-store` + `X-Accel-Buffering: no` | kept; `no-transform` added (alvyn) |
| 14 | Multi-topic subscribe (comma-separated globs) | `?topic=a.*,b.*`, hub-side fnmatch filter |
| 15 | `ready` hello event on connect with server info | `emit_sse` sends `event: ready` first |
| 16 | In-process publish API (Mercure "publish to hub") | `EventHub.inject(topic, payload)` — ad-hoc live pushes without waiting for the next poll |
| 17 | Context-manager server, `subscriber_count` | `StreamServer.__enter__/__exit__`, property |
| 18 | Specific error types | `SubscriberLimitExceeded`, `StreamClosed` in `errors.py` |

Deliberately NOT taken: JWT/auth (the API server already gates `/stream`
behind capabilities — auth stays there); async rewrite (module is
stdlib-only sync by design; thread-per-connection is kept but the *polling*
is centralized so the bottleneck the load-testing guide warns about is
gone); compression knobs (we simply never compress, per sse-starlette).
