<?php
/**
 * Devon_AJAX — the bridge between the browser and Devon's API.
 *
 * The browser never calls Devon's API directly. It calls
 * WordPress (admin-ajax.php) with a nonce, and these handlers
 * call Devon's API server-side, with the owner token, and hand
 * the answer back. That is why CORS is never needed and the
 * token never leaves the server.
 *
 * Who may call what:
 * - Chat and status: anyone who can see the shortcode (the
 *   owner chooses where to put it). Both logged-in and logged-out
 *   visitors are accepted — the nonce is only rendered with the
 *   shortcode, so the endpoint is reachable exactly where the
 *   owner embedded the interface.
 * - Pairing state and pairing actions: admins only
 *   (manage_options). Pairing is owner-only on Devon's side
 *   (SYS_CONFIG); the WordPress side enforces the same.
 *
 * Every handler verifies the nonce first, then the capability
 * where one applies, then calls the API and returns the API's
 * own answer — or the API's own error, in plain words.
 */

declare(strict_types=1);

if (!defined('ABSPATH')) {
    exit;
}

class Devon_AJAX
{
    public function register(): void
    {
        add_action('wp_ajax_devon_chat', [$this, 'chat']);
        add_action('wp_ajax_nopriv_devon_chat', [$this, 'chat']);
        add_action('wp_ajax_devon_status', [$this, 'status']);
        add_action('wp_ajax_nopriv_devon_status', [$this, 'status']);
        add_action('wp_ajax_devon_pair_state', [$this, 'pair_state']);
        add_action('wp_ajax_devon_pair_action', [$this, 'pair_action']);
    }

    private function api_or_error(): ?Devon_API
    {
        $api = Devon_API::from_settings();
        if ($api === null) {
            wp_send_json_error([
                'message' => 'Devon is not configured yet — set the API base URL and the API token in Settings → Devon Interface.',
            ]);
            return null;
        }
        return $api;
    }

    /**
     * POST a chat turn. Body (JSON): {messages: [{role, content}…]}.
     * Sends the whole conversation to POST /chat and returns the
     * API's own response: text, model, provider, degraded,
     * fallback_note, error, latency_ms.
     */
    public function chat(): void
    {
        check_ajax_referer('devon_interface', 'nonce');
        $api = $this->api_or_error();
        if ($api === null) {
            return;
        }
        $raw = isset($_POST['messages']) ? wp_unslash($_POST['messages']) : '';
        $messages = json_decode(is_string($raw) ? $raw : '', true);
        if (!is_array($messages)) {
            $messages = [];
        }
        $result = $api->chat($messages);
        if (!$result['ok']) {
            wp_send_json_error(['message' => $result['error']]);
            return;
        }
        $data = $result['data'];
        if (!empty($data['error'])) {
            wp_send_json_error(['message' => (string) $data['error']]);
            return;
        }
        wp_send_json_success([
            'text' => (string) ($data['text'] ?? ''),
            'model' => (string) ($data['model'] ?? ''),
            'provider' => (string) ($data['provider'] ?? ''),
            'degraded' => !empty($data['degraded']),
            'fallback_note' => (string) ($data['fallback_note'] ?? ''),
            'latency_ms' => (float) ($data['latency_ms'] ?? 0),
        ]);
    }

    /**
     * GET the status: /live (is she answering?), /health (version,
     * schema, uptime, providers, runtime state), and the pairing
     * counts (registered / connected / dropped). Each leg is
     * reported on its own — a bridge being down does not hide
     * the health, and the health being down does not hide the
     * liveness.
     */
    public function status(): void
    {
        check_ajax_referer('devon_interface', 'nonce');
        $api = $this->api_or_error();
        if ($api === null) {
            return;
        }
        $live = $api->live();
        $health = $api->health();
        $pair = $api->pair_state();

        $out = [
            'live' => $live['ok'],
            'live_error' => $live['ok'] ? '' : $live['error'],
            'health' => null,
            'health_error' => $health['ok'] ? '' : $health['error'],
            'counts' => null,
            'counts_error' => $pair['ok'] ? '' : $pair['error'],
        ];
        if ($health['ok'] && is_array($health['data'])) {
            $out['health'] = [
                'version' => (string) ($health['data']['version'] ?? ''),
                'schema_version' => (int) ($health['data']['schema_version'] ?? 0),
                'pid' => (int) ($health['data']['pid'] ?? 0),
                'uptime_s' => (float) ($health['data']['uptime_s'] ?? 0),
                'providers' => array_values((array) ($health['data']['providers'] ?? [])),
                'runtime' => is_array($health['data']['runtime'] ?? null)
                    ? $health['data']['runtime']
                    : ['state' => 'unknown'],
            ];
        }
        if ($pair['ok'] && is_array($pair['data'])) {
            $out['counts'] = is_array($pair['data']['counts'] ?? null)
                ? $pair['data']['counts']
                : null;
            $out['owner_stopped'] = is_array($pair['data']['owner_stopped'] ?? null)
                ? $pair['data']['owner_stopped']
                : null;
        }
        wp_send_json_success($out);
    }

    /**
     * GET the pairing pool state. Admins only.
     */
    public function pair_state(): void
    {
        check_ajax_referer('devon_interface', 'nonce');
        if (!current_user_can('manage_options')) {
            wp_send_json_error(['message' => 'Pairing is owner-only.']);
            return;
        }
        $api = $this->api_or_error();
        if ($api === null) {
            return;
        }
        $result = $api->pair_state();
        if (!$result['ok']) {
            wp_send_json_error(['message' => $result['error']]);
            return;
        }
        // Same inner-ok rule as pair_action: the contract can
        // answer ok:false inside a 200 body.
        $data = $result['data'];
        if (is_array($data) && array_key_exists('ok', $data) && $data['ok'] === false) {
            wp_send_json_error(['message' => (string) ($data['error'] ?? 'the pairing state failed')]);
            return;
        }
        wp_send_json_success($data);
    }

    /**
     * POST one pairing action. Body (JSON): the action payload —
     * {action, id?, number?, label?, name?, on?, code?, password?}.
     * The payload is passed to POST /pair/action as-is (after the
     * nonce and the admin check), and the API's own answer comes
     * back — including the fresh state, so the page re-renders
     * the truth in one round trip.
     */
    public function pair_action(): void
    {
        check_ajax_referer('devon_interface', 'nonce');
        if (!current_user_can('manage_options')) {
            wp_send_json_error(['message' => 'Pairing is owner-only.']);
            return;
        }
        $api = $this->api_or_error();
        if ($api === null) {
            return;
        }
        $raw = isset($_POST['payload']) ? wp_unslash($_POST['payload']) : '';
        $payload = json_decode(is_string($raw) ? $raw : '', true);
        if (!is_array($payload)) {
            $payload = [];
        }
        $result = $api->pair_action($payload);
        if (!$result['ok']) {
            wp_send_json_error(['message' => $result['error']]);
            return;
        }
        // The pairing contract answers failures as a 200 with
        // ok:false inside the body (the API's own words, not an
        // HTTP error) — the inner flag is the verdict, not the
        // status line. A failed action must not read as done.
        $data = $result['data'];
        if (is_array($data) && array_key_exists('ok', $data) && $data['ok'] === false) {
            wp_send_json_error(['message' => (string) ($data['error'] ?? 'the action failed')]);
            return;
        }
        wp_send_json_success($data);
    }
}
