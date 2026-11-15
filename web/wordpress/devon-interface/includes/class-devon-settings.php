<?php
/**
 * Devon_Settings — the Settings API page (Settings → Devon Interface).
 *
 * Two settings, both entered by the owner in wp-admin, never baked
 * into the code:
 * - api_base_url — where Devon serves her API (e.g. the EC2 address
 *   and port, http://...:8787). Trailing slashes are trimmed.
 * - api_token    — the owner API token (the one with full power).
 *   Stored as a plain option, like any WordPress credential the
 *   owner types in; it is never printed back and never sent to
 *   the browser.
 *
 * The page also carries a "Test connection" that calls GET /live
 * (no auth needed) and GET /health (needs the token) and reports
 * exactly what Devon answered — so a wrong address or a wrong
 * token is visible here, before any shortcode is embedded.
 */

declare(strict_types=1);

if (!defined('ABSPATH')) {
    exit;
}

class Devon_Settings
{
    public function register(): void
    {
        add_action('admin_menu', [$this, 'add_menu']);
        add_action('admin_init', [$this, 'register_settings']);
    }

    public function add_menu(): void
    {
        add_options_page(
            __('Devon Interface', 'devon-interface'),
            __('Devon Interface', 'devon-interface'),
            'manage_options',
            'devon-interface',
            [$this, 'render_page']
        );
    }

    public function register_settings(): void
    {
        register_setting(
            'devon_interface',
            DEVON_INTERFACE_OPTION,
            [
                'type' => 'array',
                'sanitize_callback' => [$this, 'sanitize'],
                'default' => ['api_base_url' => '', 'api_token' => ''],
            ]
        );

        add_settings_section(
            'devon_interface_connection',
            __('Connection to Devon', 'devon-interface'),
            [$this, 'render_section'],
            'devon-interface'
        );

        add_settings_field(
            'api_base_url',
            __('API base URL', 'devon-interface'),
            [$this, 'render_base_url_field'],
            'devon-interface',
            'devon_interface_connection'
        );

        add_settings_field(
            'api_token',
            __('API token', 'devon-interface'),
            [$this, 'render_token_field'],
            'devon-interface',
            'devon_interface_connection'
        );
    }

    /**
     * Sanitize: the base URL is a URL with no trailing slash; the
     * token is a plain string, kept as typed. An empty token field
     * on save keeps the stored token (so saving the URL does not
     * wipe it — the field shows a placeholder, not the value).
     */
    public function sanitize($input): array
    {
        $existing = get_option(DEVON_INTERFACE_OPTION, []);
        if (!is_array($existing)) {
            $existing = [];
        }
        $out = [
            'api_base_url' => '',
            'api_token' => (string) ($existing['api_token'] ?? ''),
        ];
        if (is_array($input)) {
            $url = trim((string) ($input['api_base_url'] ?? ''));
            if ($url !== '') {
                $out['api_base_url'] = rtrim(esc_url_raw($url), '/');
            }
            $token = trim((string) ($input['api_token'] ?? ''));
            if ($token !== '') {
                $out['api_token'] = sanitize_text_field($token);
            }
        }
        return $out;
    }

    public function render_section(): void
    {
        echo '<p>' . esc_html__(
            'Devon runs on her own host. These two settings tell WordPress where she is and how to prove it is you. ' .
            'The token is the owner API token — the one with full power. It is stored here, in the WordPress database, ' .
            'and it is never sent to the browser: every call to Devon is made server-side, from PHP.',
            'devon-interface'
        ) . '</p>';
    }

    public function render_base_url_field(): void
    {
        $settings = get_option(DEVON_INTERFACE_OPTION, []);
        $value = is_array($settings) ? (string) ($settings['api_base_url'] ?? '') : '';
        echo '<input type="url" name="' . esc_attr(DEVON_INTERFACE_OPTION) . '[api_base_url]" ' .
            'value="' . esc_attr($value) . '" class="regular-text" ' .
            'placeholder="http://your-ec2-address:8787" autocomplete="off">';
        echo '<p class="description">' . esc_html__(
            'Where Devon serves her API. The default port is 8787. Use the address your EC2 box answers on — ' .
            'WordPress must be able to reach it over the network.',
            'devon-interface'
        ) . '</p>';
    }

    public function render_token_field(): void
    {
        $settings = get_option(DEVON_INTERFACE_OPTION, []);
        $has_token = is_array($settings) && !empty($settings['api_token']);
        echo '<input type="password" name="' . esc_attr(DEVON_INTERFACE_OPTION) . '[api_token]" ' .
            'value="" class="regular-text" placeholder="' .
            ($has_token ? esc_attr__('token saved — leave blank to keep it', 'devon-interface') : esc_attr__('paste the owner API token', 'devon-interface')) .
            '" autocomplete="off">';
        echo '<p class="description">' . esc_html__(
            'The owner API token (NM_API_TOKEN on Devon\'s host). Leave it blank when saving to keep the stored token. ' .
            'It is never shown back and never leaves the server.',
            'devon-interface'
        ) . '</p>';
    }

    /**
     * The settings page, with a connection test that reports
     * exactly what Devon answered: /live (no auth) and /health
     * (needs the token). A wrong address or a wrong token shows
     * here, in plain words, before any shortcode is embedded.
     */
    public function render_page(): void
    {
        if (!current_user_can('manage_options')) {
            wp_die(esc_html__('You do not have permission to access this page.', 'devon-interface'));
        }

        $test = null;
        if (isset($_GET['devon_test']) && $_GET['devon_test'] === '1') {
            check_admin_referer('devon_interface_test');
            $test = $this->test_connection();
        }

        echo '<div class="wrap">';
        echo '<h1>' . esc_html__('Devon Interface', 'devon-interface') . '</h1>';
        echo '<p>' . esc_html__(
            'Devon\'s web interface on WordPress. Chat with her brain, see her live status, manage her paired accounts — ' .
            'all through her API, which stays on her own host.',
            'devon-interface'
        ) . '</p>';

        if (is_array($test)) {
            $this->render_test_result($test);
        }

        echo '<form method="post" action="options.php">';
        settings_fields('devon_interface');
        do_settings_sections('devon-interface');
        submit_button();
        echo '</form>';

        $test_url = wp_nonce_url(
            add_query_arg('devon_test', '1', menu_page_url('devon-interface', false)),
            'devon_interface_test'
        );
        echo '<h2>' . esc_html__('Connection', 'devon-interface') . '</h2>';
        echo '<p><a class="button" href="' . esc_url($test_url) . '">' .
            esc_html__('Test connection', 'devon-interface') . '</a></p>';
        echo '<p class="description">' . esc_html__(
            'Calls GET /live (no auth) and GET /health (needs the token) and reports what Devon answered.',
            'devon-interface'
        ) . '</p>';

        echo '<h2>' . esc_html__('Shortcodes', 'devon-interface') . '</h2>';
        echo '<p>' . esc_html__(
            'Embed the interface in any page or post:',
            'devon-interface'
        ) . '</p>';
        echo '<ul>';
        echo '<li><code>[devon_chat]</code> — ' . esc_html__('chat with Devon\'s brain', 'devon-interface') . '</li>';
        echo '<li><code>[devon_status]</code> — ' . esc_html__('her live status', 'devon-interface') . '</li>';
        echo '<li><code>[devon_pairing]</code> — ' . esc_html__('paired accounts (admins only)', 'devon-interface') . '</li>';
        echo '<li><code>[devon_interface]</code> — ' . esc_html__('all three, on one page', 'devon-interface') . '</li>';
        echo '</ul>';
        echo '</div>';
    }

    /**
     * Test the connection: /live first (proves the address
     * answers), then /health (proves the token is accepted).
     * Returns ['live' => ..., 'health' => ...] envelopes from
     * Devon_API::request.
     */
    private function test_connection(): array
    {
        $api = Devon_API::from_settings();
        if ($api === null) {
            return [
                'live' => ['ok' => false, 'status' => 0, 'data' => null,
                    'error' => 'not configured yet — set the API base URL and the API token above, then save.'],
                'health' => ['ok' => false, 'status' => 0, 'data' => null, 'error' => ''],
            ];
        }
        return [
            'live' => $api->live(),
            'health' => $api->health(),
        ];
    }

    private function render_test_result(array $test): void
    {
        $live = $test['live'];
        $health = $test['health'];
        echo '<div class="notice notice-' . ($live['ok'] ? 'success' : 'error') . '"><p><strong>' .
            esc_html__('GET /live: ', 'devon-interface') . '</strong>';
        if ($live['ok']) {
            echo esc_html__('Devon answered — the address is right.', 'devon-interface');
        } else {
            echo esc_html($live['error']);
        }
        echo '</p></div>';
        echo '<div class="notice notice-' . ($health['ok'] ? 'success' : 'error') . '"><p><strong>' .
            esc_html__('GET /health: ', 'devon-interface') . '</strong>';
        if ($health['ok']) {
            $version = (string) ($health['data']['version'] ?? '');
            $runtime = (string) (($health['data']['runtime']['state'] ?? ''));
            echo esc_html__('Devon answered', 'devon-interface');
            if ($version !== '') {
                echo ' — ' . esc_html__('version ', 'devon-interface') . esc_html($version);
            }
            if ($runtime !== '') {
                echo ' — ' . esc_html__('runtime ', 'devon-interface') . esc_html($runtime);
            }
            echo '.';
        } else {
            echo esc_html($health['error']);
        }
        echo '</p></div>';
    }
}
