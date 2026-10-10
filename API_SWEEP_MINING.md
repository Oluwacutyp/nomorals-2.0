# API Sweep — External Mining (2026-10-10)

Module: `nomorals/api/` — `server.py` (APIServer), `mcp_server.py` (MCPServer), `acp.py` (ACPServer).
Method: mine best-in-class implementations OUTSIDE this repo first, then build the gold in.

## 1. APIServer (stdlib HTTP API) — mined gold

| # | Gold | Source | What we take |
|---|------|--------|--------------|
| 1 | **RFC 9457 Problem Details** — `application/problem+json`, fields `type`/`title`/`status`/`detail`/`instance`, extension members permitted | RFC 9457 (obsoletes RFC 7807, March 2024); real adoptions: github.com/gongahkia/tanabata#20, github.com/lgtm-hq/podex#235, github.com/open-analysi/analysi-app (UnifiedAPIResponses.md), github.com/howells/agentsurface (error-handling.md), UK gov DWT API standards | Replace bare `{"error": msg}` bodies with problem-details envelopes; `instance` = request id; keep `error` + `code` as extension members for backward compat |
| 2 | **Request IDs** — `x-request-id` on every response; `requestId` in error bodies; used to correlate with logs | UK gov DWT API standards doc; GOV.UK API standards; Caddy `request_id` | Honor client-sent `X-Request-ID`, else mint `req_<id>`; echo on every response; put in problem body `instance`; log lines carry it |
| 3 | **Token-bucket rate limiting, per key, O(1) state** — the industry default for public APIs: AWS API Gateway, Stripe use it; bursts allowed up to capacity; two numbers per key | github.com/krishpatel10/rate-limiter (algorithm comparison: token bucket = AWS/Stripe; sliding log = exactness for login/OTP), dev.to/rogo032 (rate-limiter design series: token bucket w/ small bucket recommended), dev.to/logrocket (TokenBucket impl pattern) | In-process token bucket keyed by principal name; lazy refill on `time.monotonic()`; configurable per-principal limits |
| 4 | **429 etiquette: `Retry-After` + standing headers** — RFC 6585 §4: 429 MAY carry `Retry-After`; GitHub sends `x-ratelimit-limit/-remaining/-used/-reset` on EVERY response; IETF draft standardizing `RateLimit`/`RateLimit-Policy` | dev.to/rogo032 (cites RFC 6585, RFC 9110, GitHub headers, IETF draft v11) | Emit `X-RateLimit-Limit/Remaining/Reset` on all responses when limiting is on; `Retry-After` (seconds) on 429 |
| 5 | **Liveness ≠ readiness** — `/live` = "process alive" (static, tokenless); `/ready` = "can serve traffic" (checks DB) | Kubernetes probe semantics (universal convention) | Add auth'd `GET /ready` doing a real DB check → 200 or 503 problem |
| 6 | **HTTP hardening middleware** — max request bytes, inbound rate limits, sensitive-header guards | github.com/nullablevariant/rust-mcp-core (`http_hardening` feature: `max_request_bytes`, inbound rate limits, session abuse controls, panic/sensitive-header guards) | Security response headers (`X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`); rate limiter above; existing body caps stay |
| 7 | **CORS for API servers** — configurable origins, OPTIONS preflight with `Access-Control-Allow-*` | Universal web-API practice | `cors_origins` constructor arg; `do_OPTIONS` preflight; origin echo on responses only when configured (default: no CORS headers — no behavior change) |
| 8 | **405 for wrong method on a known path** — method mismatch is not "not found" | Universal REST practice | `dispatch` returns 405 + `Allow` list when path is registered under other methods |

## 2. MCPServer — mined gold

All from the official spec and SDKs (spec: https://modelcontextprotocol.io, protocol version 2025-11-25; SDK: https://github.com/modelcontextprotocol/python-sdk):

| # | Gold | Source | What we take |
|---|------|--------|--------------|
| 1 | **`ToolAnnotations`**: `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint` + `title` | Official python-sdk README (`mcp.types.ToolAnnotations`); dev.to/kamelyoul MCP-server article (2026-10) | Annotate all 4 tools truthfully: `memory_query` read-only/idempotent; `memory_write` destructive; `tool_call` destructive + open-world; `research_run` read-only + open-world |
| 2 | **Structured output**: `outputSchema` on the tool + `structuredContent` in `CallToolResult`, with unstructured `content` kept for backward compat | Official python-sdk README ("Structured results are automatically validated… For backward compatibility, unstructured results are also returned") | Add `outputSchema` to memory_query/memory_write/research_run; return both `content` (text) and `structuredContent` |
| 3 | **`completion/complete`** — server declares `completions` capability; params `ref` (`ref/prompt`|`ref/resource`) + `argument` (`name`,`value`); returns `completion: {values, total, hasMore}`; SDK enforces 100-item limit; server validates the referenced prompt/resource exists | Official mcp-ruby-sdk README (maintained with Shopify); github.com/jamesward/zio-http-mcp SPEC.md (cross-cutting pagination/completion) | Implement `completion/complete` for prompt-arg (`devon-brief`/`topic`) and resource-uri completion; advertise `completions: {}` |
| 4 | **`logging/setLevel` + `notifications/message`** — server capability `logging`; client sets level, server pushes structured log notifications | rust-mcp-core `client_logging` feature; jamesward SPEC.md capability table (`logging` — emits structured log messages) | Implement `logging/setLevel` (validated level names), thread-local emit hook, `_notify()` gated by level; stdio `emit` writes real notification frames |
| 5 | **`notifications/progress`** — request carries `_meta.progressToken` (string\|int); server MAY send `notifications/progress` with `progress` (MUST increase), optional `total`, `message`; MUST stop after completion | Official spec page modelcontextprotocol.io/specification/2025-11-25/basic/utilities/progress; jamesward SPEC.md ("Any request can include `_meta.progressToken`") | Honor `progressToken` on `tools/call`: emit progress=1 at start, progress=2 at end (monotone, stops after completion) |
| 6 | **`resources/templates/list`** — URI templates alongside static resources; unknown resource → `-32002` (Resource not found) | jamesward SPEC.md capability/error-code tables | Add `devon://memory/facts/{id}` template; route templated reads to the two-tier fact store; `-32002` on miss |
| 7 | **Protocol version negotiation** — server responds with highest mutually-supported version | AwareXOne agentic-bug-hunter research brief (2026-09-17, from official sources): stable revision `2026-07-28`, legacy `2025-11-25`+ with initialize handshake | Negotiate across known versions `(2025-11-25, 2025-06-18, 2025-03-26, 2024-11-05)`; echo the highest both support, never above our max |
| 8 | **Unknown-tool-name error shape** — `isError: true` with actionable text | awarexone brief ("Tool errors should use `isError: true` with actionable content") | Already the shape; keep |

Honestly NOT built (stay -32601, documented): `sampling/*`, `elicitation/*` (server→client, no transport story in buffered mode), `roots/*` (client-side), task-augmented `tasks/*` (experimental), `resources/subscribe`.

## 3. ACPServer — mined gold

All verified against the official v1 schema: `github.com/zed-industries/agent-client-protocol` → `schema/v1/schema.json` (downloaded and parsed 2026-10-10):

| # | Gold | Source (schema def) | What we take |
|---|------|---------------------|--------------|
| 1 | **`ToolCall.kind`** enum: `read edit delete move search execute think fetch switch_mode other` — "helps clients choose appropriate icons and UI treatment" | `$defs/ToolKind` | Map registry tool names → kinds by name heuristics (read_/get_/list_→read, write_/edit_→edit, delete_→delete, search_/query_→search, run_/exec_/shell_→execute, web_/http_→fetch…); default `other` |
| 2 | **`ToolCall.name`** (programmatic tool name) + **`rawInput`** (raw input params) on the initial `tool_call` notification; **`rawOutput`** on updates | `$defs/ToolCall` (name/rawInput/rawOutput fields) | traced_call now sends name, kind, title, rawInput at creation; rawOutput on completion |
| 3 | **`ToolCallLocation`** — `{path, line?}` "enables follow-along features in clients" | `$defs/ToolCallLocation` | Extract absolute-path args (`path`/`file`/`filepath`) from tool kwargs → `locations` on the notification |
| 4 | **`plan` session update** — "Agents report plans to clients to provide visibility into their execution strategy"; client replaces the whole plan per update | `$defs/Plan`, `$defs/PlanEntry` (content/priority/status: pending/in_progress/completed) | Emit a plan update at turn start (1 entry, in_progress, the real goal text) and at turn end (completed); full replacement each time, per spec |
| 5 | **`available_commands_update`** — "Available commands are ready or have changed" | `$defs/SessionUpdate` oneOf + `$defs/AvailableCommand` (name/description required) | Emit on `session/new` with capability-filtered registry tools as commands |
| 6 | **`agentCapabilities.mcpCapabilities`** `{http, sse}` — advertised MCP support | `$defs/AgentCapabilities`, `$defs/McpCapabilities` | Advertise `{http:false, sse:false}` (honest: declared mcpServers are accepted, not wired) |
| 7 | **Client capabilities on `initialize`** — client advertises `fs.{readTextFile,writeTextFile}`, `terminal`, `auth`, `elicitation` | `$defs/ClientCapabilities`, `$defs/InitializeRequest` | Record `clientCapabilities` on the server at initialize (real data; enables future agent→client fs/terminal calls) |
| 8 | **`session/prompt` request shape** — required: `sessionId`, `prompt` only; response: `stopReason` only | `$defs/PromptRequest`, `$defs/PromptResponse` | Confirmed current implementation is spec-true; no change needed |
| 9 | **Stop reasons** `end_turn|max_tokens|max_turn_requests|refusal|cancelled`; "Custom or future stop reasons MUST begin with `_`" | `$defs/StopReason`; v2 migration doc | Keep; internal `__CANCELLED__`/`__REFUSAL__` markers are server-side only, never on the wire |
| 10 | **`mode_update`/`usage_update`** session updates exist | degnbol/agentic.nvim ACP SKILL.md (from official schema) | Skip `usage_update` (no token accounting in the loop); keep `session/set_mode` (already implemented) |

Honestly NOT built (stay -32601, documented): `session/request_permission` (agent→client; our capability grants deny-by-default instead), `fs/*`, `terminal/*` (client-side role), elicitation (client must advertise `elicitation` capability — recorded, not yet exercised).

## 4. What stays untouched

- The stdlib-only, no-dependency stance of `server.py` (works on a phone with nothing installed).
- The Principal/capability model — the same auth strength across HTTP, MCP, ACP.
- `session/request_permission` non-implementation (documented, principled: capability grants deny by default).
- Sampling/elicitation over MCP (server→client needs a request/response transport story we don't have in buffered/stdio modes).
