<?php
/**
 * Devon's own pairing page (§15) — the PHP web rung.
 *
 * One file, no framework.  Served by Devon on loopback with PHP's
 * built-in server (`php -S 127.0.0.1:<port> web/pairing.php`).
 * The page owns NOTHING: every render is Devon's pairing contract
 * read fresh (GET /pair/state on Devon's API), and every action
 * posts to that same contract (POST /pair/action) with the
 * owner's API token.  Actions therefore land in the same
 * registry / supervisor the chat /pair flow drives — the page
 * and the chat can never drift into two pools.
 *
 * Gates, in order: the page token Devon generated for this
 * workspace (DEVON_PAIR_PAGE_TOKEN) — without it configured the
 * page refuses everything, and knowing it is what logs you in;
 * then Devon's own API gate (the sys.config capability behind
 * DEVON_API_TOKEN) on every contract call.  When the API is
 * down or refuses, the page says so — it never invents a pool.
 *
 * Approval still only ever happens on the phone: WhatsApp →
 * Linked Devices → Link with phone number, or the login code
 * Telegram sends the account itself.  This page shows codes and
 * drives sessions; it cannot approve one.
 */
declare(strict_types=1);

$API_BASE  = rtrim(getenv('DEVON_API_BASE') ?: 'http://127.0.0.1:8787', '/');
$API_TOKEN = (string)(getenv('DEVON_API_TOKEN') ?: '');
$PAGE_TOKEN = (string)(getenv('DEVON_PAIR_PAGE_TOKEN') ?: '');

session_set_cookie_params(['httponly' => true, 'samesite' => 'Strict', 'path' => '/']);
session_start();
if (empty($_SESSION['csrf'])) {
    $_SESSION['csrf'] = bin2hex(random_bytes(16));
}

function h($v): string { return htmlspecialchars((string)$v, ENT_QUOTES, 'UTF-8'); }

function api_call(string $method, string $path, ?array $payload = null): array
{
    global $API_BASE, $API_TOKEN;
    $ch = curl_init($API_BASE . $path);
    $headers = ['Accept: application/json'];
    if ($API_TOKEN !== '') {
        $headers[] = 'Authorization: Bearer ' . $API_TOKEN;
    }
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => $method === 'POST' ? 60 : 15,
        CURLOPT_CONNECTTIMEOUT => 3,
        CURLOPT_HTTPHEADER => $headers,
    ]);
    if ($method === 'POST') {
        curl_setopt($ch, CURLOPT_POST, true);
        $headers[] = 'Content-Type: application/json';
        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload ?? new stdClass()));
    }
    $raw = curl_exec($ch);
    if ($raw === false) {
        $err = curl_error($ch);
        curl_close($ch);
        return ['_transport' => false, 'error' => "Devon's API is not reachable at $API_BASE ($err) — is Devon running?"];
    }
    $status = (int)curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
    curl_close($ch);
    $data = json_decode((string)$raw, true);
    if (!is_array($data)) {
        return ['_transport' => false, 'error' => "Devon's API answered HTTP $status with a non-JSON body."];
    }
    $data['_status'] = $status;
    $data['_transport'] = true;
    return $data;
}

function api_state(): array
{
    $s = api_call('GET', '/pair/state');
    if (($s['_transport'] ?? false) === false) {
        return ['ok' => false, 'unavailable' => $s['error']];
    }
    if (($s['_status'] ?? 0) === 403) {
        return ['ok' => false, 'unavailable' =>
            "Devon's API refused the pairing contract (403). The contract is owner-only: " .
            "the page must be launched by Devon with the owner API token configured " .
            "(NM_API_TOKEN). No pool state is shown without it."];
    }
    if (($s['_status'] ?? 0) !== 200) {
        return ['ok' => false, 'unavailable' =>
            "Devon's API answered HTTP " . ($s['_status'] ?? '?') . " for the pairing state."];
    }
    return $s;
}

function authed(): bool { return !empty($_SESSION['owner']); }

$path = parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH) ?: '/';
$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';

// The page is not armed without its gate token: refuse honestly.
if ($PAGE_TOKEN === '') {
    http_response_code(503);
    header('Content-Type: text/plain; charset=utf-8');
    echo "This pairing page is not armed: Devon launches it with a page token " .
         "(DEVON_PAIR_PAGE_TOKEN). Start it from chat with /pair page.\n";
    exit;
}

if ($path === '/logout' && $method === 'POST') {
    $_SESSION = [];
    session_destroy();
    header('Location: /login');
    exit;
}

if ($path === '/login') {
    if ($method === 'POST') {
        $given = (string)($_POST['token'] ?? '');
        $csrf = (string)($_POST['csrf'] ?? '');
        if (!hash_equals($_SESSION['csrf'], $csrf)) {
            $loginError = 'stale form — try again';
        } elseif (hash_equals($PAGE_TOKEN, $given)) {
            session_regenerate_id(true);
            $_SESSION['owner'] = true;
            header('Location: /');
            exit;
        } else {
            $loginError = 'that is not the page token';
        }
    }
    if (authed()) { header('Location: /'); exit; }
    header('Content-Type: text/html; charset=utf-8');
    ?><!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Devon — pairing page</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:26rem;padding:0 1rem}
input,button{font-size:1rem;padding:.5rem;width:100%;margin:.4rem 0;box-sizing:border-box}
.err{color:#b00020}</style></head><body>
<h1>📟 Devon pairing page</h1>
<p>This page controls your paired accounts. It opens with the page
token Devon gave you in chat (<code>/pair page</code>).</p>
<?php if (!empty($loginError)): ?><p class="err"><?= h($loginError) ?></p><?php endif; ?>
<form method="post" action="/login">
<input type="hidden" name="csrf" value="<?= h($_SESSION['csrf']) ?>">
<input type="password" name="token" placeholder="page token" autocomplete="off" autofocus>
<button type="submit">Open the pool</button>
</form></body></html><?php
    exit;
}

if (!authed()) {
    if ($path === '/state.json') {
        http_response_code(401);
        header('Content-Type: application/json');
        echo json_encode(['ok' => false, 'error' => 'not logged in']);
        exit;
    }
    header('Location: /login');
    exit;
}

if ($path === '/state.json') {
    $s = api_state();
    header('Content-Type: application/json');
    echo json_encode($s);
    exit;
}

if ($path === '/action' && $method === 'POST') {
    $csrf = (string)($_POST['csrf'] ?? '');
    if (!hash_equals($_SESSION['csrf'], $csrf)) {
        header('Location: /?flash=' . urlencode('⚠️ stale form (csrf) — nothing was done'));
        exit;
    }
    $action = (string)($_POST['action'] ?? '');
    $payload = ['action' => $action];
    foreach (['id', 'number', 'label', 'name', 'code', 'password'] as $k) {
        if (isset($_POST[$k])) $payload[$k] = (string)$_POST[$k];
    }
    if (isset($_POST['on'])) $payload['on'] = ((string)$_POST['on'] === '1');
    if ($action === 'remove') {
        $payload['confirm'] = ((string)($_POST['confirm'] ?? '') === '1');
    }
    $r = api_call('POST', '/pair/action', $payload);
    if (($r['_transport'] ?? false) === false) {
        $msg = '⚠️ ' . $r['error'];
    } elseif (($r['_status'] ?? 0) === 403) {
        $msg = "⚠️ Devon's API refused the action (403) — the pairing contract is owner-only.";
    } elseif (!empty($r['ok'])) {
        $msg = '✅ ' . (string)($r['note'] ?? ($action . ' done'));
    } else {
        $msg = '⚠️ ' . (string)($r['error'] ?? 'the action failed');
    }
    header('Location: /?flash=' . urlencode($msg));
    exit;
}

if ($path !== '/') {
    http_response_code(404);
    header('Content-Type: text/plain; charset=utf-8');
    echo "not here — the pairing page lives at /\n";
    exit;
}

$state = api_state();
$flash = (string)($_GET['flash'] ?? '');
header('Content-Type: text/html; charset=utf-8');
?><!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Devon — paired accounts</title>
<style>
body{font-family:system-ui,sans-serif;margin:1.5rem auto;max-width:72rem;padding:0 1rem;color:#111}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(21rem,1fr));gap:1rem}
.card{border:1px solid #ccc;border-radius:.6rem;padding:1rem}
.card h2{margin:.1rem 0 .4rem;font-size:1.05rem;word-break:break-all}
.chip{display:inline-block;border:1px solid #999;border-radius:1rem;padding:.05rem .5rem;margin:.1rem;font-size:.8rem}
.chip.on{background:#e2f5e2}.chip.off{opacity:.65}.chip.warn{background:#fde8e8;border-color:#c0392b;color:#7b1d12}
.code{font-family:ui-monospace,monospace;font-size:1.9rem;letter-spacing:.15em;background:#111;color:#ffe98a;
 padding:.6rem;border-radius:.5rem;text-align:center;margin:.5rem 0}
button{cursor:pointer} .row form{display:inline-block;margin:.1rem}
.err{background:#fdecec;border:1px solid #b00020;padding:.8rem;border-radius:.5rem}
.warn{background:#fff7e0;border:1px solid #b58900;padding:.8rem;border-radius:.5rem}
.flash{background:#eef5ff;border:1px solid #557;padding:.7rem;border-radius:.5rem;margin-bottom:1rem}
fieldset{border:1px solid #ccc;border-radius:.6rem;margin-bottom:1rem}
input{padding:.35rem} .muted{opacity:.7;font-size:.9rem}
</style></head><body>
<h1>📟 Paired accounts</h1>
<?php if ($flash !== ''): ?><div class="flash"><?= h($flash) ?></div><?php endif; ?>
<?php if (empty($state['ok'])): ?>
<div class="err"><?= h($state['unavailable'] ?? 'pool state unavailable') ?></div>
<?php else: ?>
<?php $c = $state['counts']; ?>
<p>Registered: <b><?= h($c['registered']) ?></b> · Connected: <b><?= h($c['connected']) ?></b> ·
Dropped: <b><?= h($c['dropped']) ?></b>
<span class="muted">· the same pool /pair shows in chat — one registry, every surface</span></p>
<?php if (!empty($state['owner_stopped']['whatsapp'])): ?>
<div class="warn">⏸️ WhatsApp is under an owner stop (recorded in chat or here).
Starting or pairing an account lifts it — Devon never lifts it on its own.</div>
<?php endif; ?>

<fieldset><legend>Pair a new account</legend>
<form method="post" action="/action" style="display:inline-block;margin-right:2rem">
<input type="hidden" name="csrf" value="<?= h($_SESSION['csrf']) ?>">
<input type="hidden" name="action" value="pair_whatsapp">
<label>WhatsApp number <input name="number" placeholder="2348012345678" required></label>
<label>label <input name="label" placeholder="optional"></label>
<button type="submit">Pair WhatsApp — get the 8-char code</button>
</form>
<form method="post" action="/action" style="display:inline-block">
<input type="hidden" name="csrf" value="<?= h($_SESSION['csrf']) ?>">
<input type="hidden" name="action" value="pair_telegram">
<label>Telegram number <input name="number" placeholder="2348012345678" required></label>
<label>label <input name="label" placeholder="optional"></label>
<button type="submit">Pair Telegram — it sends a login code</button>
</form>
<p class="muted">Pairing is approved on the phone, always: WhatsApp → Linked Devices →
Link with phone number, or the login code Telegram sends the account itself.
A web page can show a code — it can never approve one.</p>
</fieldset>

<div class="grid">
<?php foreach ($state['accounts'] as $a): ?>
<div class="card">
<h2><?= h($a['id']) ?><?= $a['label'] !== '' ? ' “' . h($a['label']) . '”' : '' ?></h2>
<p><?= h($a['platform']) ?> · status: <b><?= h($a['status']) ?></b> · bridge: <?= h($a['process']) ?>
<?php if (!empty($a['primary'])): ?> · 🎯 default for <?= h($a['platform']) ?><?php endif; ?></p>
<?php if (!empty($a['pairing_code'])): ?>
<div class="code" data-code-for="<?= h($a['id']) ?>"><?= h($a['pairing_code']['code']) ?></div>
<p class="muted">pairing code, <?= h((int)$a['pairing_code']['age_s']) ?>s old — type it into
Linked Devices NOW; codes expire in minutes.</p>
<?php endif; ?>
<?php if (!empty($a['pairing_error'])): ?><p class="err">⚠️ <?= h($a['pairing_error']) ?></p><?php endif; ?>
<?php if (!empty($a['last_error']) && empty($a['pairing_error'])): ?><p class="muted">⚠️ <?= h($a['last_error']) ?></p><?php endif; ?>
<?php if (!empty($a['tg_flow'])): ?>
<p>⏳ Telegram login code sent (<?= h($a['tg_flow']['state']) ?>,
<?= h($a['tg_flow']['expires_in_s']) ?>s left) — answer it here:</p>
<form method="post" action="/action">
<input type="hidden" name="csrf" value="<?= h($_SESSION['csrf']) ?>">
<input type="hidden" name="action" value="tgcode">
<input type="hidden" name="id" value="<?= h($a['id']) ?>">
<input name="code" placeholder="login code" required>
<button type="submit">Send code</button>
</form>
<form method="post" action="/action">
<input type="hidden" name="csrf" value="<?= h($_SESSION['csrf']) ?>">
<input type="hidden" name="action" value="tgpass">
<input type="hidden" name="id" value="<?= h($a['id']) ?>">
<input name="password" type="password" placeholder="2FA password (only if asked)">
<button type="submit">Send password</button>
</form>
<?php endif; ?>
<p><?php foreach ($a['toggles'] as $name => $on): ?>
<span class="chip <?= $on ? 'on' : 'off' ?>"><?= h($name) ?>: <?= $on ? 'on' : 'off' ?></span>
<?php endforeach; ?></p>
<?php if (!empty($a['live'])): ?>
<p class="live-line"><?php foreach ($a['live'] as $name => $verdict): ?>
<span class="chip <?= str_starts_with($verdict, '✓') ? 'on' : 'warn' ?>"><?= h($name) ?>: <?= h($verdict) ?></span>
<?php endforeach; ?></p>
<?php endif; ?>
<div class="row">
<?php foreach ($a['toggles'] as $name => $on): ?>
<form method="post" action="/action">
<input type="hidden" name="csrf" value="<?= h($_SESSION['csrf']) ?>">
<input type="hidden" name="action" value="toggle">
<input type="hidden" name="id" value="<?= h($a['id']) ?>">
<input type="hidden" name="name" value="<?= h($name) ?>">
<input type="hidden" name="on" value="<?= $on ? '0' : '1' ?>">
<button type="submit"><?= h($name) ?> → <?= $on ? 'off' : 'on' ?></button>
</form>
<?php endforeach; ?>
</div>
<div class="row">
<?php foreach (['start', 'stop', 'restart'] as $verb): ?>
<form method="post" action="/action">
<input type="hidden" name="csrf" value="<?= h($_SESSION['csrf']) ?>">
<input type="hidden" name="action" value="<?= $verb ?>">
<input type="hidden" name="id" value="<?= h($a['id']) ?>">
<button type="submit"><?= $verb ?></button>
</form>
<?php endforeach; ?>
<?php if (empty($a['primary'])): ?>
<form method="post" action="/action">
<input type="hidden" name="csrf" value="<?= h($_SESSION['csrf']) ?>">
<input type="hidden" name="action" value="use">
<input type="hidden" name="id" value="<?= h($a['id']) ?>">
<button type="submit">make default</button>
</form>
<?php endif; ?>
<form method="post" action="/action" onsubmit="return confirm('Remove <?= h($a['id']) ?> from the pool? Its saved session stays on disk.')">
<input type="hidden" name="csrf" value="<?= h($_SESSION['csrf']) ?>">
<input type="hidden" name="action" value="remove">
<input type="hidden" name="id" value="<?= h($a['id']) ?>">
<input type="hidden" name="confirm" value="1">
<button type="submit">remove</button>
</form>
</div>
</div>
<?php endforeach; ?>
</div>
<?php if (empty($state['accounts'])): ?>
<p>No accounts paired yet — pair one above, or in chat: <code>/pair whatsapp &lt;number&gt;</code>.</p>
<?php endif; ?>
<?php endif; ?>
<form method="post" action="/logout" style="margin-top:2rem">
<button type="submit">Log out</button>
</form>
<script>
// Live-keep: while a pairing is in flight the code lands on its own —
// poll the state and reload when the visible truth changes (never
// while the owner is typing into a form).
(function(){
  var snap = null;
  function sig(s){
    if(!s || !s.accounts) return '';
    return JSON.stringify([s.counts, s.accounts.map(function(a){
      return [a.id, a.status, a.process,
              a.pairing_code ? a.pairing_code.code : '',
              a.tg_flow ? a.tg_flow.state : ''];})]);
  }
  setInterval(function(){
    var el = document.activeElement;
    if(el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) return;
    fetch('/state.json').then(function(r){return r.json();}).then(function(s){
      var now = sig(s);
      if(snap === null){ snap = now; return; }
      if(now !== snap){ location.reload(); }
    }).catch(function(){});
  }, 3000);
})();
</script>
</body></html>
