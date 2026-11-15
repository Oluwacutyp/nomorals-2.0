/* Devon Interface — browser side.
 *
 * The browser never calls Devon's API. It calls WordPress
 * (admin-ajax.php) with a nonce, and WordPress calls Devon
 * server-side. That is why no CORS setup is needed and the API
 * token never leaves the server.
 *
 * Three views, one file:
 * - chat:    keeps the conversation in the browser, sends the
 *            whole history to POST /chat each turn (the API is
 *            stateless), renders the reply with its model,
 *            provider, and any degradation note.
 * - status:  polls the health + counts, renders them.
 * - pairing: polls the pool state, renders accounts, live
 *            pairing codes, Telegram login flows, toggles, and
 *            lifecycle actions — the same contract as Devon's
 *            own pairing page (web/pairing.php).
 */

(function () {
    'use strict';

    if (typeof devonInterface === 'undefined') {
        return;
    }

    var ajaxUrl = devonInterface.ajaxUrl;
    var nonce = devonInterface.nonce;

    function esc(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function post(action, extra) {
        var body = new FormData();
        body.append('action', action);
        body.append('nonce', nonce);
        if (extra) {
            Object.keys(extra).forEach(function (k) {
                body.append(k, extra[k]);
            });
        }
        return fetch(ajaxUrl, { method: 'POST', body: body, credentials: 'same-origin' })
            .then(function (r) { return r.json(); });
    }

    function showError(el, msg) {
        if (!el) return;
        el.hidden = false;
        el.textContent = msg;
    }

    function hideError(el) {
        if (!el) return;
        el.hidden = true;
        el.textContent = '';
    }

    /* ── chat ─────────────────────────────────────────────── */

    document.querySelectorAll('[data-devon-view="chat"]').forEach(function (root) {
        var log = root.querySelector('.devon-chat-log');
        var form = root.querySelector('.devon-chat-form');
        var input = root.querySelector('.devon-chat-input');
        var sendBtn = root.querySelector('.devon-chat-send');
        var errEl = root.querySelector('.devon-chat-error');
        var messages = [];

        function addMsg(role, text, meta) {
            var div = document.createElement('div');
            div.className = 'devon-msg devon-msg-' + (role === 'user' ? 'user' : 'devon');
            var textEl = document.createElement('div');
            textEl.textContent = text;
            div.appendChild(textEl);
            if (meta) {
                var metaEl = document.createElement('span');
                metaEl.className = 'devon-msg-meta';
                metaEl.textContent = meta;
                div.appendChild(metaEl);
            }
            log.appendChild(div);
            log.scrollTop = log.scrollHeight;
        }

        form.addEventListener('submit', function (e) {
            e.preventDefault();
            var text = input.value.trim();
            if (!text) return;
            hideError(errEl);
            messages.push({ role: 'user', content: text });
            addMsg('user', text);
            input.value = '';
            input.disabled = true;
            sendBtn.disabled = true;
            post('devon_chat', { messages: JSON.stringify(messages) })
                .then(function (resp) {
                    if (!resp.success) {
                        showError(errEl, (resp.data && resp.data.message) || 'the chat failed');
                        messages.pop();
                        return;
                    }
                    var d = resp.data || {};
                    messages.push({ role: 'assistant', content: d.text || '' });
                    var meta = [];
                    if (d.model) meta.push(d.model);
                    if (d.provider) meta.push(d.provider);
                    if (d.latency_ms) meta.push(Math.round(d.latency_ms) + 'ms');
                    if (d.degraded) {
                        meta.push('degraded' + (d.fallback_note ? ' — ' + d.fallback_note : ''));
                    }
                    addMsg('devon', d.text || '(no reply)', meta.join(' · ') || null);
                })
                .catch(function (err) {
                    showError(errEl, 'WordPress could not reach Devon (' + err.message + ') — is Devon running?');
                    messages.pop();
                })
                .finally(function () {
                    input.disabled = false;
                    sendBtn.disabled = false;
                    input.focus();
                });
        });
    });

    /* ── status ───────────────────────────────────────────── */

    function renderStatus(root, data) {
        var body = root.querySelector('.devon-status-body');
        if (!body) return;
        var html = '<div class="devon-status-grid">';

        html += '<div class="devon-status-card"><h3>Liveness</h3>';
        if (data.live) {
            html += '<p><span class="devon-chip on">answering</span></p>';
        } else {
            html += '<p><span class="devon-chip warn">not answering</span></p>';
            if (data.live_error) html += '<p>' + esc(data.live_error) + '</p>';
        }
        html += '</div>';

        html += '<div class="devon-status-card"><h3>Health</h3>';
        if (data.health) {
            var h = data.health;
            html += '<p>version <b>' + esc(h.version) + '</b></p>';
            html += '<p>schema <b>' + esc(h.schema_version) + '</b> · pid <b>' + esc(h.pid) + '</b></p>';
            html += '<p>uptime <b>' + esc(Math.round(h.uptime_s)) + 's</b></p>';
            var rt = h.runtime || {};
            var state = String(rt.state || 'unknown');
            var cls = state === 'alive' ? 'on' : (state === 'stopped' || state === 'stale' ? 'warn' : 'off');
            html += '<p>runtime <span class="devon-chip ' + cls + '">' + esc(state) + '</span></p>';
            if (rt.beacon_age_s != null) html += '<p>beacon <b>' + esc(rt.beacon_age_s) + 's</b> old</p>';
            if (rt.last_error) html += '<p>' + esc(rt.last_error) + '</p>';
            if (h.providers && h.providers.length) {
                html += '<p>providers: ' + h.providers.map(function (p) { return esc(p); }).join(', ') + '</p>';
            }
        } else {
            html += '<p><span class="devon-chip warn">unavailable</span></p>';
            if (data.health_error) html += '<p>' + esc(data.health_error) + '</p>';
        }
        html += '</div>';

        html += '<div class="devon-status-card"><h3>Accounts</h3>';
        if (data.counts) {
            var c = data.counts;
            html += '<p>registered <b>' + esc(c.registered) + '</b> · connected <b>' + esc(c.connected) + '</b> · dropped <b>' + esc(c.dropped) + '</b></p>';
        } else {
            html += '<p><span class="devon-chip warn">counts unavailable</span></p>';
            if (data.counts_error) html += '<p>' + esc(data.counts_error) + '</p>';
        }
        if (data.owner_stopped) {
            if (data.owner_stopped.whatsapp) html += '<p>WhatsApp is under an owner stop.</p>';
            if (data.owner_stopped.telegram) html += '<p>Telegram is under an owner stop.</p>';
        }
        html += '</div>';
        html += '</div>';
        body.innerHTML = html;
    }

    document.querySelectorAll('[data-devon-view="status"]').forEach(function (root) {
        var errEl = root.querySelector('.devon-status-error');
        function refresh() {
            post('devon_status', {})
                .then(function (resp) {
                    if (!resp.success) {
                        showError(errEl, (resp.data && resp.data.message) || 'the status check failed');
                        return;
                    }
                    hideError(errEl);
                    renderStatus(root, resp.data || {});
                })
                .catch(function (err) {
                    showError(errEl, 'WordPress could not reach Devon (' + err.message + ')');
                });
        }
        refresh();
        setInterval(refresh, 30000);
    });

    /* ── pairing ──────────────────────────────────────────── */

    function pairStateSig(s) {
        if (!s || !s.accounts) return '';
        return JSON.stringify([s.counts, s.accounts.map(function (a) {
            return [a.id, a.status, a.process,
                a.pairing_code ? a.pairing_code.code : '',
                a.tg_flow ? a.tg_flow.state : ''];
        })]);
    }

    function renderPairing(root, state, flashMsg) {
        var body = root.querySelector('.devon-pairing-body');
        if (!body) return;
        var html = '';
        if (flashMsg) {
            var ok = flashMsg.indexOf('✅') === 0;
            html += '<div class="devon-flash ' + (ok ? 'ok' : 'warn') + '">' + esc(flashMsg) + '</div>';
        }
        if (!state.ok) {
            html += '<p class="devon-notice">' + esc(state.unavailable || 'pool state unavailable') + '</p>';
            body.innerHTML = html;
            return;
        }
        var c = state.counts || {};
        html += '<p class="devon-counts">Registered: <b>' + esc(c.registered) + '</b> · Connected: <b>' +
            esc(c.connected) + '</b> · Dropped: <b>' + esc(c.dropped) + '</b>' +
            '<span class="muted"> · the same pool /pair shows in chat — one registry, every surface</span></p>';
        if (state.owner_stopped && state.owner_stopped.whatsapp) {
            html += '<p class="warn">WhatsApp is under an owner stop (recorded in chat or here). ' +
                'Starting or pairing an account lifts it — Devon never lifts it on her own.</p>';
        }

        html += '<fieldset><legend>Pair a new account</legend>' +
            '<form data-devon-pair-form data-action="pair_whatsapp">' +
            '<label>WhatsApp number <input name="number" placeholder="2348012345678" required></label> ' +
            '<label>label <input name="label" placeholder="optional"></label> ' +
            '<button type="submit">Pair WhatsApp — get the 8-char code</button></form>' +
            '<form data-devon-pair-form data-action="pair_telegram">' +
            '<label>Telegram number <input name="number" placeholder="2348012345678" required></label> ' +
            '<label>label <input name="label" placeholder="optional"></label> ' +
            '<button type="submit">Pair Telegram — it sends a login code</button></form>' +
            '<p class="muted">Pairing is approved on the phone, always: WhatsApp → Linked Devices → ' +
            'Link with phone number, or the login code Telegram sends the account itself. ' +
            'A web page can show a code — it can never approve one.</p></fieldset>';

        html += '<div class="devon-pair-grid">';
        (state.accounts || []).forEach(function (a) {
            var pstate = a.pairing_error ? 'failed'
                : ((a.pairing_code || a.tg_flow) ? 'waiting'
                : (a.status === 'pairing' ? 'working' : 'settled'));
            var waitMsg = '';
            if (pstate === 'waiting') {
                waitMsg = a.tg_flow ? 'waiting for the login code answer below'
                    : 'waiting for this code to be typed on the phone';
            } else if (pstate === 'working') {
                waitMsg = 'pairing in flight — the bridge is working';
            }
            html += '<div class="devon-pair-card" data-pair-state="' + esc(pstate) + '">';
            html += '<h3>' + esc(a.id) + (a.label ? ' “' + esc(a.label) + '”' : '') + '</h3>';
            html += '<p>' + esc(a.platform) + ' · status: <b>' + esc(a.status) + '</b> · bridge: ' + esc(a.process) +
                (a.primary ? ' · 🎯 default for ' + esc(a.platform) : '') + '</p>';
            if (waitMsg) {
                html += '<p class="devon-waitrow"><span class="devon-loader" aria-hidden="true"><i></i><i></i><i></i></span>' + esc(waitMsg) + '</p>';
            }
            if (a.pairing_code) {
                html += '<div class="devon-pair-code">' + esc(a.pairing_code.code) + '</div>';
                html += '<p class="muted">pairing code, ' + esc(a.pairing_code.age_s) + 's old — type it into Linked Devices NOW; codes expire in minutes.</p>';
            }
            if (a.pairing_error) html += '<p class="err">⚠️ ' + esc(a.pairing_error) + '</p>';
            if (a.last_error && !a.pairing_error) html += '<p class="muted">⚠️ ' + esc(a.last_error) + '</p>';
            if (a.tg_flow) {
                html += '<p>⏳ Telegram login code sent (' + esc(a.tg_flow.state) + ', ' +
                    esc(a.tg_flow.expires_in_s) + 's left) — answer it here:</p>';
                html += '<form data-devon-pair-form data-action="tgcode">' +
                    '<input type="hidden" name="id" value="' + esc(a.id) + '">' +
                    '<label>login code <input name="code" placeholder="login code" autocomplete="one-time-code" required></label> ' +
                    '<button type="submit">Send code</button></form>';
                html += '<form data-devon-pair-form data-action="tgpass">' +
                    '<input type="hidden" name="id" value="' + esc(a.id) + '">' +
                    '<label>2FA password <input name="password" type="password" placeholder="2FA password (only if asked)" autocomplete="current-password"></label> ' +
                    '<button type="submit">Send password</button></form>';
            }
            html += '<p>';
            Object.keys(a.toggles || {}).forEach(function (name) {
                var on = !!a.toggles[name];
                html += '<span class="devon-chip ' + (on ? 'on' : 'off') + '">' + esc(name) + ': ' + (on ? 'on' : 'off') + '</span>';
            });
            html += '</p>';
            if (a.live) {
                html += '<p>';
                Object.keys(a.live).forEach(function (name) {
                    var verdict = String(a.live[name]);
                    var good = verdict.indexOf('✓') === 0;
                    html += '<span class="devon-chip ' + (good ? 'on' : 'warn') + '">' + esc(name) + ': ' + esc(verdict) + '</span>';
                });
                html += '</p>';
            }
            html += '<div>';
            Object.keys(a.toggles || {}).forEach(function (name) {
                var on = !!a.toggles[name];
                html += '<button type="button" class="devon-switch" role="switch" aria-checked="' + (on ? 'true' : 'false') +
                    '" data-devon-toggle data-id="' + esc(a.id) + '" data-name="' + esc(name) + '" data-on="' + (on ? '0' : '1') +
                    '" title="turn ' + esc(name) + ' ' + (on ? 'off' : 'on') + '">' +
                    '<span class="trk" aria-hidden="true"></span>' + esc(name) + '</button>';
            });
            html += '</div><div>';
            ['start', 'stop', 'restart'].forEach(function (verb) {
                html += '<form data-devon-pair-form data-action="' + verb + '">' +
                    '<input type="hidden" name="id" value="' + esc(a.id) + '">' +
                    '<button type="submit">' + verb + '</button></form>';
            });
            if (!a.primary) {
                html += '<form data-devon-pair-form data-action="use">' +
                    '<input type="hidden" name="id" value="' + esc(a.id) + '">' +
                    '<button type="submit">make default</button></form>';
            }
            html += '<form data-devon-pair-form data-action="remove" data-confirm="Remove ' + esc(a.id) + ' from the pool? Its saved session stays on disk.">' +
                '<input type="hidden" name="id" value="' + esc(a.id) + '">' +
                '<input type="hidden" name="confirm" value="1">' +
                '<button type="submit">remove</button></form>';
            html += '</div></div>';
        });
        html += '</div>';
        if (!(state.accounts || []).length) {
            html += '<p>No accounts paired yet — pair one above, or in chat: <code>/pair whatsapp &lt;number&gt;</code>.</p>';
        }
        body.innerHTML = html;
    }

    document.querySelectorAll('[data-devon-view="pairing"]').forEach(function (root) {
        var errEl = root.querySelector('.devon-pairing-error');
        var lastSig = null;
        var lastState = null;
        var flashMsg = '';

        function refresh(force) {
            post('devon_pair_state', {})
                .then(function (resp) {
                    if (!resp.success) {
                        showError(errEl, (resp.data && resp.data.message) || 'the pairing state failed');
                        return;
                    }
                    hideError(errEl);
                    var state = resp.data || {};
                    var sig = pairStateSig(state);
                    if (force || lastSig === null || sig !== lastSig || flashMsg) {
                        renderPairing(root, state, flashMsg);
                        flashMsg = '';
                    }
                    lastSig = sig;
                    lastState = state;
                })
                .catch(function (err) {
                    showError(errEl, 'WordPress could not reach Devon (' + err.message + ')');
                });
        }

        function doAction(payload) {
            hideError(errEl);
            return post('devon_pair_action', { payload: JSON.stringify(payload) })
                .then(function (resp) {
                    if (!resp.success) {
                        flashMsg = '⚠️ ' + ((resp.data && resp.data.message) || 'the action failed');
                    } else {
                        var d = resp.data || {};
                        flashMsg = '✅ ' + (d.note || ((d.action || 'action') + ' done'));
                    }
                    refresh(true);
                })
                .catch(function (err) {
                    flashMsg = '⚠️ WordPress could not reach Devon (' + err.message + ')';
                    refresh(true);
                });
        }

        root.addEventListener('submit', function (e) {
            var form = e.target.closest('[data-devon-pair-form]');
            if (!form) return;
            e.preventDefault();
            var confirmMsg = form.getAttribute('data-confirm');
            if (confirmMsg && !window.confirm(confirmMsg)) return;
            var payload = { action: form.getAttribute('data-action') || '' };
            new FormData(form).forEach(function (v, k) {
                payload[k] = v;
            });
            if (payload.on !== undefined) {
                payload.on = payload.on === '1' || payload.on === 'true';
            }
            doAction(payload);
        });

        root.addEventListener('click', function (e) {
            var btn = e.target.closest('[data-devon-toggle]');
            if (!btn) return;
            doAction({
                action: 'toggle',
                id: btn.getAttribute('data-id') || '',
                name: btn.getAttribute('data-name') || '',
                on: btn.getAttribute('data-on') === '1',
            });
        });

        refresh(true);
        setInterval(function () {
            var el = document.activeElement;
            if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) return;
            refresh(false);
        }, 5000);
    });
})();
