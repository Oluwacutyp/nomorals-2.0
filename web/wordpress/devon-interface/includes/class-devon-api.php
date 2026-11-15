<?php
/**
 * Devon_API — the server-side client for Devon's API.
 *
 * Every call is made from PHP, with the owner's API token in the
 * Authorization header. The token never reaches the browser: the
 * browser talks to WordPress AJAX, WordPress talks to Devon.
 *
 * This mirrors Devon's own pairing page (web/pairing.php): the
 * same endpoints, the same timeouts, the same honest errors.
 * A transport failure, a 403, or a non-JSON body is reported as
 * exactly that — never papered over with invented data.
 *
 * Endpoints used (all read from nomorals/api/server.py):
 * - GET  /live        — tokenless liveness probe, {"ok": true}
 * - GET  /ready       — readiness probe, {"ok": true} when the DB answers
 * - GET  /health      — version, schema, pid, uptime, providers, runtime
 * - POST /chat        — chat with the model router (MODEL_CALL)
 * - GET  /pair/state  — pairing pool state (SYS_CONFIG)
 * - POST /pair/action — one pairing pool action (SYS_CONFIG)
 * - GET  /docs        — self-documentation: every route on the API
 */

declare(strict_types=1);

if (!defined('ABSPATH')) {
    exit;
}

class Devon_API
{
    private string $base_url;
    private string $token;

    public function __construct(string $base_url, string $token)
    {
        $this->base_url = rtrim(trim($base_url), '/');
        $this->token = trim($token);
    }

    /**
     * Build a client from the stored settings. Returns null when
     * the plugin is not configured yet (no base URL or no token).
     */
    public static function from_settings(): ?self
    {
        $settings = get_option(DEVON_INTERFACE_OPTION, []);
        $base_url = is_array($settings) ? (string) ($settings['api_base_url'] ?? '') : '';
        $token = is_array($settings) ? (string) ($settings['api_token'] ?? '') : '';
        if ($base_url === '' || $token === '') {
            return null;
        }
        return new self($base_url, $token);
    }

    public function is_configured(): bool
    {
        return $this->base_url !== '' && $this->token !== '';
    }

    public function base_url(): string
    {
        return $this->base_url;
    }

    /**
     * One call to Devon's API.
     *
     * Returns an array with, always:
     * - 'ok'        bool — true only on a 2xx response with a JSON body
     * - 'status'    int  — the HTTP status (0 when nothing answered)
     * - 'data'      mixed — the decoded JSON body (null when unusable)
     * - 'error'     string — plain words when 'ok' is false, else ''
     *
     * The shape mirrors pairing.php's api_call: '_transport' is
     * folded into 'ok'/'error' here because WordPress callers
     * read one envelope, not two.
     */
    public function request(string $method, string $path, ?array $payload = null): array
    {
        $url = $this->base_url . $path;
        $headers = [
            'Accept' => 'application/json',
            'Authorization' => 'Bearer ' . $this->token,
        ];
        $args = [
            'headers' => $headers,
            'timeout' => $method === 'POST' ? 60 : 15,
            'connect_timeout' => 3,
        ];
        if ($method === 'POST') {
            $args['method'] = 'POST';
            $args['headers']['Content-Type'] = 'application/json';
            $args['body'] = wp_json_encode($payload ?? new stdClass());
            $response = wp_remote_post($url, $args);
        } else {
            $response = wp_remote_get($url, $args);
        }

        if (is_wp_error($response)) {
            return [
                'ok' => false,
                'status' => 0,
                'data' => null,
                'error' => "Devon's API is not reachable at {$this->base_url} (" .
                    $response->get_error_message() .
                    ') — is Devon running?',
            ];
        }

        $status = (int) wp_remote_retrieve_response_code($response);
        $raw = (string) wp_remote_retrieve_body($response);
        $data = json_decode($raw, true);
        if (!is_array($data)) {
            return [
                'ok' => false,
                'status' => $status,
                'data' => null,
                'error' => "Devon's API answered HTTP {$status} with a non-JSON body.",
            ];
        }

        if ($status === 403) {
            return [
                'ok' => false,
                'status' => $status,
                'data' => $data,
                'error' => "Devon's API refused this call (403). The pairing " .
                    'contract and the health probe are owner-only: the API ' .
                    'token must be the owner token (the one with full power). ' .
                    'No state is shown without it.',
            ];
        }

        if ($status < 200 || $status >= 300) {
            $detail = (string) ($data['detail'] ?? $data['error'] ?? '');
            return [
                'ok' => false,
                'status' => $status,
                'data' => $data,
                'error' => "Devon's API answered HTTP {$status} for {$path}" .
                    ($detail !== '' ? ": {$detail}" : '.'),
            ];
        }

        return [
            'ok' => true,
            'status' => $status,
            'data' => $data,
            'error' => '',
        ];
    }

    /**
     * GET /live — tokenless liveness probe. True when Devon answers.
     * Used by the settings page's "test connection" and by the
     * status view's first line. No auth, no DB, no version facts.
     */
    public function live(): array
    {
        return $this->request('GET', '/live');
    }

    /**
     * GET /ready — readiness probe. 200 when the database answers.
     */
    public function ready(): array
    {
        return $this->request('GET', '/ready');
    }

    /**
     * GET /health — version, schema_version, pid, uptime_s,
     * providers, and runtime {state, uptime_s, beacon_age_s,
     * last_error}. The runtime state is one of alive / stopped /
     * stale / not_running / unknown.
     */
    public function health(): array
    {
        return $this->request('GET', '/health');
    }

    /**
     * GET /docs — every route on the API, with descriptions.
     * Used by the settings page to prove the token reaches the
     * real API (and to show what the plugin can call).
     */
    public function docs(): array
    {
        return $this->request('GET', '/docs');
    }

    /**
     * POST /chat — chat with Devon's brain (her model router).
     *
     * $messages is the conversation so far: a list of
     * ['role' => 'user'|'assistant', 'content' => string]. The
     * API is stateless — the caller keeps the history and sends
     * it whole each turn.
     *
     * Returns the API's own response: on success the data holds
     * LLMResponse.to_dict() — text, model, provider, usage,
     * finish_reason, latency_ms, error, degraded, failed_providers,
     * fallback_note, failure_class, route_trace, cost_usd,
     * adaptation_note, deferred. The reply is 'text'; 'degraded'
     * and 'fallback_note' say when a fallback provider served it.
     */
    public function chat(array $messages, float $temperature = 0.7, int $max_tokens = 1024): array
    {
        $clean = [];
        foreach ($messages as $m) {
            if (!is_array($m)) {
                continue;
            }
            $role = (string) ($m['role'] ?? 'user');
            if ($role !== 'user' && $role !== 'assistant' && $role !== 'system') {
                $role = 'user';
            }
            $content = trim((string) ($m['content'] ?? ''));
            if ($content === '') {
                continue;
            }
            $clean[] = ['role' => $role, 'content' => $content];
        }
        if (empty($clean)) {
            return [
                'ok' => false,
                'status' => 0,
                'data' => null,
                'error' => 'nothing to send — the message is empty.',
            ];
        }
        return $this->request('POST', '/chat', [
            'messages' => $clean,
            'temperature' => $temperature,
            'max_tokens' => $max_tokens,
        ]);
    }

    /**
     * GET /pair/state — the pairing pool as data: counts,
     * accounts (with toggle states, live verdicts, live pairing
     * codes, pending Telegram flows), owner_stopped, the toggle
     * names, and the surface rungs. Counts are read without
     * probing, so the state renders even when a bridge is down.
     */
    public function pair_state(): array
    {
        return $this->request('GET', '/pair/state');
    }

    /**
     * POST /pair/action — one pairing pool action. $payload is
     * ['action' => ..., ...] — the actions are pair_whatsapp,
     * pair_telegram, tgcode, tgpass, start, stop, restart, remove,
     * use, toggle. Every success carries the fresh state in
     * data['state']; failures carry data['error'] in plain words.
     */
    public function pair_action(array $payload): array
    {
        $action = trim((string) ($payload['action'] ?? ''));
        if ($action === '') {
            return [
                'ok' => false,
                'status' => 0,
                'data' => null,
                'error' => 'no pairing action given.',
            ];
        }
        return $this->request('POST', '/pair/action', $payload);
    }
}
