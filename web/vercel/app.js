/* Devon web — API client + views.
 *
 * Talks to Devon's HTTP API (nomorals/api/server.py) only. Every endpoint
 * used here is a real route on that server; nothing is invented:
 *
 *   GET  /live         tokenless liveness probe -> {"ok": true}
 *   GET  /health       version, schema, uptime, providers, runtime beacon
 *   GET  /ready        {"ok": true} when the database answers, else 503
 *   POST /chat         {messages:[{role,content}], temperature, max_tokens}
 *                      -> LLMResponse dict {text, model, provider, ...}
 *   GET  /events       {events: [...]}  (EventBus snapshot)
 *   GET  /tools        {tools: [...]}   (schemas the token may call)
 *   GET  /pair/state   {ok, counts, accounts[], owner_stopped, toggles, ...}
 *   POST /pair/action  {action, ...}    -> {ok, action, note, state} or
 *                      {ok:false, action, error}
 *
 * Auth is a Bearer token (the owner's API token, NM_API_TOKEN). The token
 * and the API address live in this browser's localStorage only — they are
 * never baked into the page and never sent anywhere except Devon's API.
 *
 * Two ways to reach the API (Settings -> mode):
 *   direct — the browser calls the API address straight. Needs the API to
 *            allow this site's origin (CORS on the API side).
 *   proxy  — the browser calls this same site under /api/..., and Vercel
 *            rewrites it to the API. No CORS needed, but the rewrite rule
 *            must be added to vercel.json first (see README).
 */
(function () {
  "use strict";

  var LS_KEY = "devon.web.settings.v1";

  /* ── settings ─────────────────────────────────────────────── */

  function loadSettings() {
    try {
      var raw = localStorage.getItem(LS_KEY);
      if (!raw) return { apiBase: "", token: "", mode: "direct" };
      var s = JSON.parse(raw);
      return {
        apiBase: String(s.apiBase || "").replace(/\/+$/, ""),
        token: String(s.token || ""),
        mode: s.mode === "proxy" ? "proxy" : "direct",
      };
    } catch (e) {
      return { apiBase: "", token: "", mode: "direct" };
    }
  }

  function saveSettings(s) {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify({
        apiBase: s.apiBase,
        token: s.token,
        mode: s.mode,
      }));
    } catch (e) { /* private mode: settings just don't persist */ }
  }

  function clearSettings() {
    try { localStorage.removeItem(LS_KEY); } catch (e) { /* ignore */ }
  }

  var settings = loadSettings();

  /* ── dom helpers ──────────────────────────────────────────── */

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }
  function esc(s) { return String(s === undefined || s === null ? "" : s); }

  var toastTimer = null;
  function toast(msg, kind) {
    var t = $("toast");
    t.textContent = msg;
    t.className = "toast" + (kind ? " " + kind : "");
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.hidden = true; }, 6000);
  }

  function fmtTime(ts) {
    if (!ts) return "—";
    try {
      var d = new Date(typeof ts === "number" && ts < 1e12 ? ts * 1000 : ts);
      return isNaN(d.getTime()) ? esc(ts) : d.toLocaleString();
    } catch (e) { return esc(ts); }
  }
  function fmtSecs(s) {
    if (s === undefined || s === null || isNaN(s)) return "—";
    s = Math.round(s);
    if (s < 60) return s + "s";
    if (s < 3600) return Math.floor(s / 60) + "m " + (s % 60) + "s";
    return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m";
  }

  /* ── api client ───────────────────────────────────────────── */

  /* A rejected fetch is a network failure (unreachable host, or the
   * browser blocking a cross-origin call — CORS). An HTTP response with
   * a status is a real answer from the API. The two are told apart here
   * so the UI can say which one it is instead of guessing. */
  function api(path, opts) {
    opts = opts || {};
    var method = opts.method || "GET";
    var base;
    if (settings.mode === "proxy") {
      base = "/api";
    } else {
      if (!settings.apiBase) {
        return Promise.reject(new Error(
          "No API address saved. Open Settings and enter Devon's API address."));
      }
      base = settings.apiBase;
    }
    var headers = { "Content-Type": "application/json" };
    if (settings.token) headers["Authorization"] = "Bearer " + settings.token;
    var init = { method: method, headers: headers };
    if (opts.body !== undefined) init.body = JSON.stringify(opts.body);
    return fetch(base + path, init).then(function (resp) {
      return resp.text().then(function (raw) {
        var data = null;
        try { data = raw ? JSON.parse(raw) : null; } catch (e) { data = null; }
        if (!resp.ok) {
          var detail = "";
          if (data) {
            detail = data.detail || data.error || data.title || "";
            if (data.code) detail = detail ? detail + " (" + data.code + ")" : data.code;
          }
          var err = new Error(detail ||
            ("Devon's API answered " + resp.status + " for " + method + " " + path));
          err.status = resp.status;
          err.http = true;
          throw err;
        }
        return data;
      });
    }).catch(function (err) {
      if (err.http || err.status) throw err;
      /* fetch itself rejected: unreachable, or the browser refused a
       * cross-origin call. In direct mode that refusal is almost always
       * CORS — the API has not been told to allow this site's origin. */
      var e = new Error(
        "Could not reach Devon's API at " +
        (settings.mode === "proxy" ? "this site's /api proxy" : settings.apiBase || "(no address set)") +
        ". " +
        (settings.mode === "direct"
          ? "If she is running, this is usually CORS: her API must allow this site's address — see the README (CORS), or switch to Vercel proxy mode in Settings."
          : "If the proxy rule is not in vercel.json yet, add it (README) and redeploy — or switch to Direct mode in Settings."));
      e.network = true;
      throw e;
    });
  }

  /* Tokenless probe: is anything answering at the address at all? */
  function probeLive() {
    return api("/live", { method: "GET" });
  }

  /* ── connection bar ───────────────────────────────────────── */

  var connState = "unknown"; /* unknown | ok | bad */

  function setConn(state, text) {
    connState = state;
    var dot = $("conn-dot");
    dot.className = "conn-dot " + state;
    $("conn-text").textContent = text;
  }

  function testConnection(showToast) {
    if (settings.mode === "direct" && !settings.apiBase) {
      setConn("bad", "No API address saved — open Settings.");
      return Promise.resolve(false);
    }
    setConn("unknown", "Checking…");
    return probeLive().then(function () {
      return api("/health").then(function (h) {
        var rt = (h && h.runtime && h.runtime.state) || "unknown";
        setConn("ok", "Connected — Devon " + (h && h.version ? "v" + h.version : "") +
          ", runtime " + rt + ", schema v" + (h && h.schema_version));
        if (showToast) toast("Connected to Devon.", "ok");
        return true;
      });
    }).catch(function (err) {
      setConn("bad", err.message);
      if (showToast) toast(err.message, "err");
      return false;
    });
  }

  /* ── routing ──────────────────────────────────────────────── */

  var VIEWS = ["chat", "status", "pair"];
  var currentView = "chat";

  function route() {
    var hash = (location.hash || "#/chat").replace(/^#\/?/, "");
    if (VIEWS.indexOf(hash) === -1) hash = "chat";
    currentView = hash;
    VIEWS.forEach(function (v) {
      var sec = $("view-" + v);
      if (sec) sec.hidden = (v !== hash);
      var link = document.querySelector('.nav-link[data-view="' + v + '"]');
      if (link) link.classList.toggle("active", v === hash);
    });
    if (hash === "status") refreshStatus();
    if (hash === "pair") refreshPair();
  }

  /* ── chat ─────────────────────────────────────────────────── */

  var chatMessages = []; /* [{role, content}] sent to POST /chat */

  function chatAdd(role, text, meta) {
    var log = $("chat-log");
    var row = el("div", "chat-row " + role);
    var who = el("div", "chat-who", role === "user" ? "You" : "Devon");
    var bubble = el("div", "chat-bubble");
    bubble.textContent = text;
    row.appendChild(who);
    row.appendChild(bubble);
    log.appendChild(row);
    if (meta) {
      var m = el("div", "chat-row-meta");
      m.textContent = meta;
      log.appendChild(m);
    }
    log.scrollTop = log.scrollHeight;
  }

  function chatMetaLine(resp) {
    var bits = [];
    if (resp.model) bits.push("model " + resp.model);
    if (resp.provider) bits.push("via " + resp.provider);
    if (resp.latency_ms !== undefined) bits.push(resp.latency_ms + " ms");
    if (resp.cost_usd) bits.push("$" + resp.cost_usd);
    if (resp.degraded) {
      bits.push("DEGRADED — " + (resp.fallback_note || "served by a fallback"));
    }
    if (resp.adaptation_note) bits.push(resp.adaptation_note);
    return bits.join(" · ");
  }

  function sendChat(text) {
    var input = $("chat-input");
    var sendBtn = $("chat-send");
    chatMessages.push({ role: "user", content: text });
    chatAdd("user", text);
    input.value = "";
    input.disabled = true;
    sendBtn.disabled = true;
    var temp = parseFloat($("chat-temp").value);
    var maxTok = parseInt($("chat-tokens").value, 10);
    var body = {
      messages: chatMessages,
      temperature: isNaN(temp) ? 0.7 : temp,
      max_tokens: isNaN(maxTok) ? 1024 : maxTok,
    };
    $("chat-meta").textContent = "Devon is thinking…";
    return api("/chat", { method: "POST", body: body }).then(function (resp) {
      resp = resp || {};
      if (resp.error) {
        chatAdd("devon", "I could not answer that: " + resp.error +
          (resp.failure_class ? " (" + resp.failure_class + ")" : ""));
        $("chat-meta").textContent = chatMetaLine(resp);
        chatMessages.pop(); /* don't keep a failed turn in context */
        return;
      }
      chatMessages.push({ role: "assistant", content: resp.text || "" });
      chatAdd("devon", resp.text || "(empty reply)", chatMetaLine(resp));
      $("chat-meta").textContent = chatMetaLine(resp);
    }).catch(function (err) {
      chatMessages.pop();
      chatAdd("devon", "That did not reach Devon: " + err.message);
      $("chat-meta").textContent = "";
    }).finally(function () {
      input.disabled = false;
      sendBtn.disabled = false;
      input.focus();
    });
  }

  /* ── status ───────────────────────────────────────────────── */

  var statusTimer = null;

  function refreshStatus() {
    var cards = $("status-cards");
    var eventsEl = $("events-list");
    var toolsEl = $("tools-list");
    var apiBase = settings.mode === "proxy" ? location.origin : settings.apiBase;
    var devonLink = $("link-devon");
    var rulesLink = $("link-rules");
    if (apiBase) {
      devonLink.href = apiBase.replace(/\/+$/, "") + "/devon";
      rulesLink.href = apiBase.replace(/\/+$/, "") + "/rules";
    } else {
      devonLink.href = "#";
      rulesLink.href = "#";
    }
    cards.innerHTML = "";
    eventsEl.innerHTML = "";
    toolsEl.innerHTML = "";
    var card = function (title, body, tone) {
      var c = el("div", "status-card" + (tone ? " " + tone : ""));
      c.appendChild(el("h3", "", title));
      if (typeof body === "string") {
        c.appendChild(el("p", "", body));
      } else {
        c.appendChild(body);
      }
      cards.appendChild(c);
      return c;
    };
    var kv = function (obj) {
      var dl = el("dl", "kv");
      Object.keys(obj).forEach(function (k) {
        var dt = el("dt", "", k);
        var dd = el("dd", "", obj[k]);
        dl.appendChild(dt);
        dl.appendChild(dd);
      });
      return dl;
    };
    var busy = card("Checking…", "Asking Devon's API.");
    var jobs = [];
    jobs.push(api("/health").then(function (h) {
      busy.remove();
      var rt = (h && h.runtime) || {};
      var tone = rt.state === "alive" ? "ok" : (rt.state === "stale" || rt.state === "stopped" ? "warn" : "");
      card("Health", kv({
        "Version": h.version || "—",
        "Schema": "v" + (h.schema_version === undefined ? "?" : h.schema_version),
        "Uptime": fmtSecs(h.uptime_s),
        "Process": h.pid ? "pid " + h.pid : "—",
        "Providers": (h.providers && h.providers.length) ? h.providers.join(", ") : "none listed",
        "Runtime": rt.state || "unknown",
        "Runtime uptime": fmtSecs(rt.uptime_s),
        "Beacon age": rt.beacon_age_s !== undefined ? rt.beacon_age_s + "s" : "—",
        "Last error": rt.last_error || "—",
      }), tone);
    }).catch(function (err) {
      busy.remove();
      card("Health", "Could not read health: " + err.message, "err");
    }));
    jobs.push(api("/ready").then(function () {
      card("Database", "Ready — the database answers.", "ok");
    }).catch(function (err) {
      card("Database", "Not ready: " + err.message, "err");
    }));
    jobs.push(api("/events").then(function (r) {
      var evs = (r && r.events) || [];
      if (!evs.length) {
        eventsEl.appendChild(el("p", "panel-note", "No recent events."));
        return;
      }
      evs.slice(-20).reverse().forEach(function (ev) {
        var row = el("div", "event-row");
        var topic = ev.topic || ev.type || "(event)";
        var when = ev.ts ? fmtTime(ev.ts) : "";
        row.appendChild(el("span", "event-topic", topic));
        if (when) row.appendChild(el("span", "event-time", when));
        var data = ev.data !== undefined ? ev.data : ev;
        var detail = "";
        if (data && typeof data === "object") {
          var keys = Object.keys(data);
          detail = keys.slice(0, 4).map(function (k) {
            return k + ": " + JSON.stringify(data[k]);
          }).join(" · ");
        } else if (data !== undefined) {
          detail = String(data);
        }
        if (detail) row.appendChild(el("span", "event-data", detail));
        eventsEl.appendChild(row);
      });
    }).catch(function (err) {
      eventsEl.appendChild(el("p", "panel-note", "Could not read events: " + err.message));
    }));
    jobs.push(api("/tools").then(function (r) {
      var tools = (r && r.tools) || [];
      if (!tools.length) {
        toolsEl.appendChild(el("p", "panel-note", "No tools listed for this token."));
        return;
      }
      tools.forEach(function (t) {
        var name = (t && (t.name || t.id)) || "tool";
        var desc = (t && (t.description || t.summary)) || "";
        var chip = el("span", "tool-chip", name + (desc ? " — " + desc : ""));
        toolsEl.appendChild(chip);
      });
    }).catch(function (err) {
      toolsEl.appendChild(el("p", "panel-note", "Could not list tools: " + err.message));
    }));
    return Promise.all(jobs.map(function (p) { return p.catch(function () {}); }));
  }

  function scheduleStatus() {
    clearInterval(statusTimer);
    statusTimer = setInterval(function () {
      if (currentView === "status") refreshStatus();
    }, 15000);
  }

  /* ── pairing ──────────────────────────────────────────────── */

  var pairState = null;

  function pairAction(action, fields) {
    var body = Object.assign({ action: action }, fields || {});
    return api("/pair/action", { method: "POST", body: body }).then(function (resp) {
      resp = resp || {};
      if (resp.state) {
        pairState = resp.state;
        renderPair(pairState);
      }
      if (resp.ok === false) {
        toast("Pairing: " + (resp.error || "failed"), "err");
        return resp;
      }
      if (resp.note) toast(resp.note, "ok");
      return resp;
    }).catch(function (err) {
      toast("Pairing action failed: " + err.message, "err");
      throw err;
    });
  }

  function toggleLabel(name) {
    return name.replace(/_/g, " ");
  }

  function renderPair(state) {
    var countsEl = $("pair-counts");
    var listEl = $("pair-accounts");
    countsEl.innerHTML = "";
    listEl.innerHTML = "";
    if (!state) return;
    var c = state.counts || {};
    countsEl.appendChild(el("span", "count-chip",
      "Registered: " + (c.registered === undefined ? "?" : c.registered)));
    countsEl.appendChild(el("span", "count-chip",
      "Connected: " + (c.connected === undefined ? "?" : c.connected)));
    countsEl.appendChild(el("span", "count-chip",
      "Dropped: " + (c.dropped === undefined ? "?" : c.dropped)));
    if (state.owner_stopped) {
      if (state.owner_stopped.whatsapp) {
        countsEl.appendChild(el("span", "count-chip warn", "WhatsApp stopped by owner"));
      }
      if (state.owner_stopped.telegram) {
        countsEl.appendChild(el("span", "count-chip warn", "Telegram stopped by owner"));
      }
    }
    var accounts = state.accounts || [];
    if (!accounts.length) {
      listEl.appendChild(el("p", "panel-note", "No accounts yet — pair one below."));
      return;
    }
    accounts.forEach(function (a) {
      var card = el("div", "account-card");
      var head = el("div", "account-head");
      head.appendChild(el("strong", "", (a.platform || "?") + " · " + (a.label || a.identity || a.id)));
      if (a.primary) head.appendChild(el("span", "chip on", "primary"));
      head.appendChild(el("span", "chip " + (a.status === "linked" ? "on" : (a.status === "down" ? "warn" : "")), a.status || "?"));
      head.appendChild(el("span", "chip", a.process || "—"));
      card.appendChild(head);
      var meta = el("div", "account-meta");
      meta.textContent = "id " + a.id +
        (a.identity ? " · " + a.identity : "") +
        (a.port ? " · port " + a.port : "") +
        (a.last_seen ? " · last seen " + fmtTime(a.last_seen) : "") +
        (a.last_error ? " · error: " + a.last_error : "");
      card.appendChild(meta);
      /* WhatsApp pairing code — a live code is a credential while it
       * lives; it is shown only here, never stored or logged. */
      if (a.pairing_code && a.pairing_code.code) {
        var codeBox = el("div", "pair-code");
        codeBox.appendChild(el("span", "pair-code-label", "Pairing code — approve it on the phone:"));
        var codeVal = el("code", "pair-code-value", a.pairing_code.code);
        codeBox.appendChild(codeVal);
        var age = el("span", "pair-code-age",
          a.pairing_code.age_s !== undefined ? "issued " + a.pairing_code.age_s + "s ago" : "");
        codeBox.appendChild(age);
        var copyBtn = el("button", "btn btn-small", "Copy");
        copyBtn.type = "button";
        copyBtn.addEventListener("click", function () {
          var done = function () { toast("Pairing code copied.", "ok"); };
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(a.pairing_code.code).then(done, function () {
              toast("Copy failed — select the code and copy it by hand.", "err");
            });
          } else {
            toast("Copy not available — select the code and copy it by hand.", "err");
          }
        });
        codeBox.appendChild(copyBtn);
        card.appendChild(codeBox);
      }
      if (a.pairing_error) {
        card.appendChild(el("p", "account-error", "Pairing error: " + a.pairing_error));
      }
      /* Telegram pending flow: login code / 2FA password. */
      if (a.tg_flow) {
        var tg = el("div", "tg-flow");
        tg.appendChild(el("p", "", "Telegram login in progress: " + (a.tg_flow.state || "?") +
          (a.tg_flow.expires_in_s ? " — expires in " + a.tg_flow.expires_in_s + "s" : "")));
        var codeForm = el("form", "tg-form");
        codeForm.autocomplete = "off";
        var codeInput = el("input", "");
        codeInput.type = "text";
        codeInput.placeholder = "Login code";
        codeInput.setAttribute("autocomplete", "one-time-code");
        codeInput.required = true;
        var codeBtn = el("button", "btn btn-small", "Send code");
        codeBtn.type = "submit";
        codeForm.appendChild(codeInput);
        codeForm.appendChild(codeBtn);
        codeForm.addEventListener("submit", function (ev) {
          ev.preventDefault();
          if (!codeInput.value) return;
          pairAction("tgcode", { id: a.id, code: codeInput.value });
          codeInput.value = "";
        });
        tg.appendChild(codeForm);
        var passForm = el("form", "tg-form");
        passForm.autocomplete = "off";
        var passInput = el("input", "");
        passInput.type = "password";
        passInput.placeholder = "2FA password (only if asked)";
        passInput.setAttribute("autocomplete", "current-password");
        var passBtn = el("button", "btn btn-small", "Send password");
        passBtn.type = "submit";
        passForm.appendChild(passInput);
        passForm.appendChild(passBtn);
        passForm.addEventListener("submit", function (ev) {
          ev.preventDefault();
          if (!passInput.value) return;
          pairAction("tgpass", { id: a.id, password: passInput.value });
          passInput.value = "";
        });
        tg.appendChild(passForm);
        card.appendChild(tg);
      }
      /* Live toggle verdicts + set-state toggles. */
      if (a.live && Object.keys(a.live).length) {
        var liveLine = el("p", "live-line");
        Object.keys(a.live).forEach(function (name) {
          var v = a.live[name];
          var ok = typeof v === "string" && v.indexOf("✓") === 0;
          liveLine.appendChild(el("span", "chip " + (ok ? "on" : "warn"), name + ": " + v));
        });
        card.appendChild(liveLine);
      }
      if (a.toggles) {
        var togRow = el("div", "row toggle-row");
        Object.keys(a.toggles).forEach(function (name) {
          var on = !!a.toggles[name];
          var btn = el("button", "switch" + (on ? " on" : ""), toggleLabel(name));
          btn.type = "button";
          btn.setAttribute("role", "switch");
          btn.setAttribute("aria-checked", on ? "true" : "false");
          btn.title = "turn " + toggleLabel(name) + " " + (on ? "off" : "on");
          btn.addEventListener("click", function () {
            btn.disabled = true;
            pairAction("toggle", { id: a.id, name: name, on: !on })
              .finally(function () { btn.disabled = false; });
          });
          togRow.appendChild(btn);
        });
        card.appendChild(togRow);
      }
      /* Lifecycle. */
      var lifeRow = el("div", "row");
      ["start", "stop", "restart", "remove", "use"].forEach(function (verb) {
        var btn = el("button", "btn btn-small", verb);
        btn.type = "button";
        if (verb === "remove") btn.classList.add("btn-danger");
        btn.addEventListener("click", function () {
          if (verb === "remove" &&
              !confirm("Remove " + (a.label || a.id) + " from the pool? Its saved session stays on disk.")) {
            return;
          }
          btn.disabled = true;
          pairAction(verb, { id: a.id }).finally(function () { btn.disabled = false; });
        });
        lifeRow.appendChild(btn);
      });
      card.appendChild(lifeRow);
      listEl.appendChild(card);
    });
  }

  function refreshPair() {
    var listEl = $("pair-accounts");
    listEl.innerHTML = "";
    listEl.appendChild(el("p", "panel-note", "Loading pairing state…"));
    api("/pair/state").then(function (state) {
      pairState = state;
      renderPair(state);
    }).catch(function (err) {
      listEl.innerHTML = "";
      listEl.appendChild(el("p", "panel-note",
        "Could not read pairing state: " + err.message +
        (err.status === 403 ? " (pairing needs the owner token — check Settings.)" : "")));
    });
  }

  /* ── settings modal ───────────────────────────────────────── */

  function openSettings() {
    $("set-api-base").value = settings.apiBase;
    $("set-token").value = settings.token;
    var radios = document.querySelectorAll('input[name="set-mode"]');
    radios.forEach(function (r) { r.checked = (r.value === settings.mode); });
    $("settings-result").textContent = "";
    $("settings-modal").hidden = false;
    $("set-api-base").focus();
  }
  function closeSettings() { $("settings-modal").hidden = true; }

  /* ── wire up ──────────────────────────────────────────────── */

  window.addEventListener("hashchange", route);

  $("chat-form").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var input = $("chat-input");
    var text = input.value.trim();
    if (!text) return;
    if (settings.mode === "direct" && !settings.apiBase) {
      toast("No API address saved — open Settings first.", "err");
      openSettings();
      return;
    }
    sendChat(text);
  });
  $("chat-clear").addEventListener("click", function () {
    chatMessages = [];
    $("chat-log").innerHTML = "";
    $("chat-meta").textContent = "";
    chatAdd("devon", "Conversation cleared. What would you like to talk about?");
  });

  $("status-refresh").addEventListener("click", function () { refreshStatus(); });
  $("pair-refresh").addEventListener("click", function () { refreshPair(); });
  $("conn-test").addEventListener("click", function () { testConnection(true); });

  $("pair-new").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var platform = $("pair-platform").value;
    var number = $("pair-number").value.trim();
    var label = $("pair-label").value.trim();
    var note = $("pair-new-note");
    if (!number) { note.textContent = "Enter the number in international format."; return; }
    note.textContent = "Starting pairing…";
    var action = platform === "telegram" ? "pair_telegram" : "pair_whatsapp";
    pairAction(action, { number: number, label: label }).then(function (resp) {
      if (resp && resp.ok !== false) {
        note.textContent = resp.note || "Pairing started — watch the account card above.";
        $("pair-number").value = "";
        $("pair-label").value = "";
      } else {
        note.textContent = (resp && resp.error) || "Pairing failed.";
      }
    }).catch(function (err) {
      note.textContent = err.message;
    });
  });

  $("settings-btn").addEventListener("click", openSettings);
  $("settings-close").addEventListener("click", closeSettings);
  $("settings-modal").addEventListener("click", function (ev) {
    if (ev.target === $("settings-modal")) closeSettings();
  });
  $("settings-form").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var apiBase = $("set-api-base").value.trim().replace(/\/+$/, "");
    var token = $("set-token").value;
    var mode = "direct";
    document.querySelectorAll('input[name="set-mode"]').forEach(function (r) {
      if (r.checked) mode = r.value;
    });
    if (mode === "direct" && !apiBase) {
      $("settings-result").textContent = "Enter the API address for Direct mode.";
      return;
    }
    settings = { apiBase: apiBase, token: token, mode: mode };
    saveSettings(settings);
    $("settings-result").textContent = "Saved.";
    closeSettings();
    testConnection(true);
    if (currentView === "status") refreshStatus();
    if (currentView === "pair") refreshPair();
  });
  $("settings-test").addEventListener("click", function () {
    var apiBase = $("set-api-base").value.trim().replace(/\/+$/, "");
    var token = $("set-token").value;
    var mode = "direct";
    document.querySelectorAll('input[name="set-mode"]').forEach(function (r) {
      if (r.checked) mode = r.value;
    });
    var prev = settings;
    settings = { apiBase: apiBase, token: token, mode: mode };
    testConnection(true).finally(function () { settings = prev; });
  });
  $("settings-clear").addEventListener("click", function () {
    clearSettings();
    settings = { apiBase: "", token: "", mode: "direct" };
    $("set-api-base").value = "";
    $("set-token").value = "";
    document.querySelectorAll('input[name="set-mode"]').forEach(function (r) {
      r.checked = (r.value === "direct");
    });
    $("settings-result").textContent = "Saved settings forgotten.";
    setConn("bad", "Not connected — open Settings and enter Devon's API address and token.");
  });

  /* ── boot ─────────────────────────────────────────────────── */

  chatAdd("devon", "Hi — I'm Devon. This page talks to me through my API. " +
    "Open Settings, enter my API address and token, then say something.");
  route();
  scheduleStatus();
  if (settings.apiBase || settings.mode === "proxy") {
    testConnection(false);
  } else {
    setConn("bad", "Not connected — open Settings and enter Devon's API address and token.");
  }
})();
