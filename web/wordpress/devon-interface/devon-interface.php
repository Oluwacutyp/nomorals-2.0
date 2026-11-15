<?php
/**
 * Plugin Name: Devon Interface
 * Plugin URI: https://github.com/Oluwacutyp/nomorals-2.0
 * Description: Devon's web interface on WordPress — chat with her brain, see her live status, and manage her paired accounts. The plugin talks to Devon's API, which stays on her own host (EC2); WordPress holds the front, Devon holds the thinking.
 * Version: 1.0.0
 * Requires at least: 6.0
 * Requires PHP: 8.1
 * Author: Devon
 * License: MIT
 * License URI: https://opensource.org/licenses/MIT
 * Text Domain: devon-interface
 *
 * Devon is a persistent personal AI agent. She runs on her own host
 * (a VM) with her database, her scheduler, her WhatsApp bridge, and
 * her Python API server. WordPress cannot host Devon herself — and
 * this plugin does not try. It renders her web interface on a
 * WordPress site by calling her API, which stays where she lives.
 *
 * Architecture (the honest shape):
 * - Every call to Devon's API is made SERVER-SIDE, from PHP, with
 *   the owner's API token. The token never reaches the browser.
 *   The browser talks only to WordPress (admin-ajax.php), and
 *   WordPress talks to Devon. This is the same pattern as Devon's
 *   own pairing page (web/pairing.php), extended into WordPress.
 * - Because the calls are server-to-server, CORS is NOT required:
 *   the browser never calls Devon's API directly, so no Origin
 *   check ever runs. The API's cors_origins setting stays off.
 * - The plugin owns NOTHING: every render is Devon's API read
 *   fresh (GET /health, GET /pair/state), and every action posts
 *   to that same API (POST /chat, POST /pair/action). If Devon's
 *   API is down or refuses, the plugin says so — it never
 *   invents a status, a reply, or a pairing code.
 *
 * Capabilities, in the order they matter:
 * - Chat: POST /chat — a real conversation with Devon's brain
 *   (her model router). The full partner turn — tools, memory,
 *   platform context — lives in Telegram/WhatsApp; this chat is
 *   her brain answering, and it says so.
 * - Status: GET /live + GET /health + the pairing counts — her
 *   version, schema, uptime, provider list, runtime state, and
 *   how many accounts are registered, connected, and dropped.
 * - Pairing: GET /pair/state + POST /pair/action — the same
 *   contract Devon's own pairing page drives: pair WhatsApp and
 *   Telegram, answer login codes, flip toggles, start/stop/
 *   restart/remove accounts, set the default. Admin-only.
 *
 * Setup: Settings → Devon Interface — the API base URL (where
 * Devon serves her API, e.g. http://127.0.0.1:8787 or your EC2
 * address) and the API token (the owner token). Then embed with
 * [devon_chat], [devon_status], [devon_pairing], or
 * [devon_interface] for all three.
 */

declare(strict_types=1);

if (!defined('ABSPATH')) {
    exit;
}

define('DEVON_INTERFACE_VERSION', '1.0.0');
define('DEVON_INTERFACE_PLUGIN_FILE', __FILE__);
define('DEVON_INTERFACE_PLUGIN_DIR', plugin_dir_path(__FILE__));
define('DEVON_INTERFACE_PLUGIN_URL', plugin_dir_url(__FILE__));
define('DEVON_INTERFACE_OPTION', 'devon_interface_settings');

require_once DEVON_INTERFACE_PLUGIN_DIR . 'includes/class-devon-api.php';
require_once DEVON_INTERFACE_PLUGIN_DIR . 'includes/class-devon-settings.php';
require_once DEVON_INTERFACE_PLUGIN_DIR . 'includes/class-devon-ajax.php';
require_once DEVON_INTERFACE_PLUGIN_DIR . 'includes/class-devon-shortcodes.php';

/**
 * Activation: record the defaults (an empty base URL and an empty
 * token) so the settings page has something to edit. Existing
 * settings are never overwritten on re-activation.
 */
function devon_interface_activate(): void
{
    $existing = get_option(DEVON_INTERFACE_OPTION, null);
    if (!is_array($existing)) {
        add_option(DEVON_INTERFACE_OPTION, [
            'api_base_url' => '',
            'api_token' => '',
        ]);
    }
}
register_activation_hook(__FILE__, 'devon_interface_activate');

/**
 * Deactivation: nothing to tear down. Settings persist so a
 * re-activation does not lose the API address and token.
 */
function devon_interface_deactivate(): void
{
    // Deliberately empty: the settings stay.
}
register_deactivation_hook(__FILE__, 'devon_interface_deactivate');

/**
 * Boot: settings page, shortcodes, and AJAX handlers.
 */
function devon_interface_init(): void
{
    $settings = new Devon_Settings();
    $settings->register();

    $shortcodes = new Devon_Shortcodes();
    $shortcodes->register();

    $ajax = new Devon_AJAX();
    $ajax->register();
}
add_action('init', 'devon_interface_init');
