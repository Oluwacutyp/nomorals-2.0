<?php
/**
 * Devon_Shortcodes — the four shortcodes that embed the interface.
 *
 * [devon_chat]      — chat with Devon's brain.
 * [devon_status]    — her live status.
 * [devon_pairing]   — paired accounts (admins only).
 * [devon_interface] — all three, on one page.
 *
 * Each shortcode renders a container the JavaScript takes over;
 * the data itself is fetched through admin-ajax.php, so the page
 * is never stale and the API token never reaches the browser.
 *
 * Assets are enqueued only on pages that actually carry one of
 * the shortcodes (checked in wp_enqueue_scripts against the
 * current post's content), and the script is localized with the
 * AJAX URL and a nonce.
 *
 * When the plugin is not configured, visitors see a plain
 * "not available" line; admins see where to configure it. The
 * configuration state itself is never leaked to visitors.
 */

declare(strict_types=1);

if (!defined('ABSPATH')) {
    exit;
}

class Devon_Shortcodes
{
    /** The shortcodes this plugin owns. */
    private const SHORTCODES = ['devon_chat', 'devon_status', 'devon_pairing', 'devon_interface'];

    public function register(): void
    {
        add_shortcode('devon_chat', [$this, 'chat']);
        add_shortcode('devon_status', [$this, 'status']);
        add_shortcode('devon_pairing', [$this, 'pairing']);
        add_shortcode('devon_interface', [$this, 'interface']);
        add_action('wp_enqueue_scripts', [$this, 'enqueue_assets']);
    }

    public function enqueue_assets(): void
    {
        if (!is_singular()) {
            return;
        }
        $post = get_post();
        if ($post === null) {
            return;
        }
        $content = (string) $post->post_content;
        $needed = false;
        foreach (self::SHORTCODES as $tag) {
            if (has_shortcode($content, $tag)) {
                $needed = true;
                break;
            }
        }
        if (!$needed) {
            return;
        }
        wp_enqueue_style(
            'devon-interface',
            DEVON_INTERFACE_PLUGIN_URL . 'assets/css/devon-interface.css',
            [],
            DEVON_INTERFACE_VERSION
        );
        wp_enqueue_script(
            'devon-interface',
            DEVON_INTERFACE_PLUGIN_URL . 'assets/js/devon-interface.js',
            [],
            DEVON_INTERFACE_VERSION,
            true
        );
        wp_localize_script('devon-interface', 'devonInterface', [
            'ajaxUrl' => admin_url('admin-ajax.php'),
            'nonce' => wp_create_nonce('devon_interface'),
            'isAdmin' => current_user_can('manage_options'),
        ]);
    }

    private function configured_notice(): string
    {
        if (Devon_API::from_settings() !== null) {
            return '';
        }
        if (current_user_can('manage_options')) {
            $url = admin_url('options-general.php?page=devon-interface');
            return '<p class="devon-notice">' .
                esc_html__('Devon is not configured yet — ', 'devon-interface') .
                '<a href="' . esc_url($url) . '">' .
                esc_html__('set the API base URL and token', 'devon-interface') .
                '</a>.</p>';
        }
        return '<p class="devon-notice">' .
            esc_html__('Devon is not available right now.', 'devon-interface') .
            '</p>';
    }

    /**
     * [devon_chat] — a conversation with Devon's brain (her model
     * router, POST /chat). The history lives in the browser and
     * is sent whole each turn; the API is stateless.
     */
    public function chat(array $atts = []): string
    {
        $notice = $this->configured_notice();
        if ($notice !== '') {
            return '<div class="devon-interface devon-chat">' . $notice . '</div>';
        }
        return '<div class="devon-interface devon-chat" data-devon-view="chat">' .
            '<div class="devon-chat-log" role="log" aria-live="polite"></div>' .
            '<p class="devon-chat-note">' .
            esc_html__('Chat with Devon\'s brain — her model router answers here. The full partner (tools, memory, platform context) lives in Telegram and WhatsApp.', 'devon-interface') .
            '</p>' .
            '<form class="devon-chat-form">' .
            '<label class="screen-reader-text" for="devon-chat-input">' . esc_html__('Message Devon', 'devon-interface') . '</label>' .
            '<input type="text" class="devon-chat-input" id="devon-chat-input" placeholder="' .
            esc_attr__('Message Devon…', 'devon-interface') . '" autocomplete="off">' .
            '<button type="submit" class="devon-chat-send">' . esc_html__('Send', 'devon-interface') . '</button>' .
            '</form>' .
            '<p class="devon-chat-error" role="alert" hidden></p>' .
            '</div>';
    }

    /**
     * [devon_status] — her live status: liveness, health
     * (version, schema, uptime, providers, runtime state), and
     * the pairing counts.
     */
    public function status(array $atts = []): string
    {
        $notice = $this->configured_notice();
        if ($notice !== '') {
            return '<div class="devon-interface devon-status">' . $notice . '</div>';
        }
        return '<div class="devon-interface devon-status" data-devon-view="status">' .
            '<div class="devon-status-body"><p class="devon-loading">' .
            esc_html__('Asking Devon…', 'devon-interface') . '</p></div>' .
            '<p class="devon-status-error" role="alert" hidden></p>' .
            '</div>';
    }

    /**
     * [devon_pairing] — the pairing pool: accounts, live pairing
     * codes, Telegram login flows, toggles, and lifecycle actions.
     * Admins only — the pairing contract is owner-only on
     * Devon's side (SYS_CONFIG), and this side enforces the same.
     */
    public function pairing(array $atts = []): string
    {
        if (!current_user_can('manage_options')) {
            return '<div class="devon-interface devon-pairing"><p class="devon-notice">' .
                esc_html__('Pairing is owner-only.', 'devon-interface') .
                '</p></div>';
        }
        $notice = $this->configured_notice();
        if ($notice !== '') {
            return '<div class="devon-interface devon-pairing">' . $notice . '</div>';
        }
        return '<div class="devon-interface devon-pairing" data-devon-view="pairing">' .
            '<div class="devon-pairing-body"><p class="devon-loading">' .
            esc_html__('Reading the pairing pool…', 'devon-interface') . '</p></div>' .
            '<p class="devon-pairing-error" role="alert" hidden></p>' .
            '</div>';
    }

    /**
     * [devon_interface] — all three views, in one place: chat,
     * status, pairing (pairing only renders for admins).
     */
    public function interface(array $atts = []): string
    {
        $out = '<div class="devon-interface-all">';
        $out .= '<section class="devon-section"><h2>' . esc_html__('Chat', 'devon-interface') . '</h2>' . $this->chat() . '</section>';
        $out .= '<section class="devon-section"><h2>' . esc_html__('Status', 'devon-interface') . '</h2>' . $this->status() . '</section>';
        $out .= '<section class="devon-section"><h2>' . esc_html__('Paired accounts', 'devon-interface') . '</h2>' . $this->pairing() . '</section>';
        $out .= '</div>';
        return $out;
    }
}
