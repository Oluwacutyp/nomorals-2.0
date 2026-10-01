"""Web console — the companion in your browser, key-gated.

``nm web`` starts a localhost HTTP server with a single page: a dark
chat console talking to the SAME partner brain that lives on the social
platforms (mood, memory, slash commands, tools — all of it).

Security model, deliberately simple:

- one randomly generated access key per boot (``secrets.token_urlsafe``),
  printed to the terminal and embedded in the URL you open
- the HTML page is a static shell with no data in it; EVERY /api/*
  call requires ``Authorization: Bearer <key>`` — no key, no chat,
  no commands, no status
- binds 127.0.0.1 by default: the port never leaves your device

Stdlib only, no external assets — the page works fully offline.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..core.logging_setup import get_logger
from ..social.chat.base import ChatKind, ChatMessage, ChatRef
from ..social.chat.gateway import ChatGateway
from ..social.chat.web import WebAdapter
from ..version import __version__

__all__ = ["WebConsole", "console_page"]

_log = get_logger(__name__)

_CHAT_TIMEOUT_FIRST = 120.0   # seconds to wait for the pump to start a reply
_QUIET = 4.0                  # seconds of silence after a part = turn complete
_HARD_CAP = 600.0             # never hold a request longer than this
_POLL_INTERVAL = 1.5          # frontend polling cadence


def _new_key() -> str:
    return secrets.token_urlsafe(12)


class WebConsole:
    """Bridges the browser to the partner runtime over localhost HTTP."""

    def __init__(self, context: Any, *, token: str = "") -> None:
        self.context = context
        self.token = token or _new_key()
        self.adapter = WebAdapter()
        from ..agents.partner_runtime import PartnerRuntime

        # A gateway with ONLY the web adapter: no social platforms start,
        # no accounts, no network peers.  The runtime itself is the full
        # partner (brain, mood, memory, control commands, tool wiring).
        self.runtime = PartnerRuntime(
            context,
            gateway=ChatGateway(
                {"web": self.adapter},
                db=context.db,
                owner_chats={"web:console"},
            ),
        )
        self.chat = self.adapter.chat
        self._msg_counter = 0
        self._msg_guard = threading.Lock()
        self.started_at = time.time()

    # ── brain bridge ─────────────────────────────────────────────────────────

    def push(self, text: str) -> str:
        """Feed one owner message into the partner. Returns its id."""
        with self._msg_guard:
            self._msg_counter += 1
            msg_id = f"web-{int(time.time() * 1000)}-{self._msg_counter}"
        self.runtime.on_message(ChatMessage(
            chat=self.chat,
            incoming=True,
            text=text,
            sender=self.chat.peer,
            message_id=msg_id,
        ))
        return msg_id

    def waiting(self) -> bool:
        """True while the runtime is still processing this chat."""
        draining = self.runtime._draining.get(self.chat.key, False)
        return bool(draining) or (
            self.adapter.pending_parts(self.chat.key) != [] and not draining)

    def collect(self, timeout: float = 10.0) -> list[str]:
        """Wait for her turn to settle, then return all captured parts.

        The pump may emit several parts (paced by human typing delays)
        and occasionally a delayed presence reply after the first pump
        completes, so "done" means: at least one part, the pump is idle,
        and no part has arrived in the last _QUIET seconds.
        """
        deadline = time.time() + min(timeout, _HARD_CAP)
        parts: list[str] = []
        while time.time() < deadline:
            parts = self.adapter.take_parts(self.chat.key)
            if parts and not self.waiting():
                # give a trailing delayed part a moment to arrive
                tail_deadline = time.time() + _QUIET
                while time.time() < tail_deadline:
                    more = self.adapter.take_parts(self.chat.key)
                    if more:
                        parts.extend(more)
                        tail_deadline = time.time() + _QUIET
                        continue
                    if self.waiting():
                        tail_deadline = time.time() + _QUIET
                        continue
                    break
                break
            time.sleep(0.15)
        else:
            parts = self.adapter.take_parts(self.chat.key)
        return parts

    def reply(self, text: str, *, timeout: float = 180.0) -> dict[str, Any]:
        self.push(text)
        parts = self.collect(timeout=timeout)
        return {
            "ok": True,
            "reply": "\n\n".join(parts),
            "parts": parts,
            "mood": self._mood_label(),
        }

    def _mood_label(self) -> str:
        try:
            return self.runtime.brain.mood.current().label
        except Exception:  # noqa: BLE001 - mood is cosmetic
            return ""

    def status(self) -> dict[str, Any]:
        brain = self.runtime.brain
        mood = brain.mood.current()
        try:
            stage = brain.stage.current().name if hasattr(brain, "stage") else ""
        except Exception:  # noqa: BLE001
            stage = ""
        provider = ""
        try:
            provider = self.context.settings.llm.provider
        except Exception:  # noqa: BLE001
            pass
        return {
            "ok": True,
            "version": __version__,
            "mood": mood.label,
            "mood_values": {k: round(v, 2) for k, v in mood.values.items()},
            "stage": stage,
            "provider": provider,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "busy": self.waiting(),
        }

    def stop(self) -> None:
        try:
            self.runtime.stop()
        except Exception:  # noqa: BLE001
            pass

    # ── HTTP ─────────────────────────────────────────────────────────────────

    def handler_class(self) -> type[BaseHTTPRequestHandler]:
        console = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = f"NoMoralsConsole/{__version__}"

            def log_message(self, fmt: str, *args: Any) -> None:
                _log.debug("console %s - %s", self.address_string(), fmt % args)

            def _send(self, status: int, payload: Any,
                      content_type: str = "application/json; charset=utf-8") -> None:
                raw = (payload if isinstance(payload, bytes)
                       else json.dumps(payload, default=str,
                                       ensure_ascii=False).encode("utf-8"))
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "null")
                self.end_headers()
                self.wfile.write(raw)

            def _authorized(self) -> bool:
                header = self.headers.get("Authorization", "")
                return header == f"Bearer {console.token}"

            def _handle(self, method: str) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/":
                    self._send(200, console_page().encode("utf-8"), "text/html; charset=utf-8")
                    return
                if path == "/health":
                    self._send(200, {"ok": True, "version": __version__})
                    return
                if not path.startswith("/api/"):
                    self._send(404, {"error": "not found"})
                    return
                if not self._authorized():
                    self._send(401, {"error": "access key required"})
                    return
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                try:
                    if method == "GET" and path == "/api/status":
                        self._send(200, console.status())
                    elif method == "POST" and path == "/api/chat":
                        length = int(self.headers.get("Content-Length") or 0)
                        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                        text = str(body.get("text") or "").strip()
                        if not text:
                            self._send(400, {"error": "text is required"})
                            return
                        try:
                            out = console.reply(text)
                        except Exception as exc:  # noqa: BLE001 - surface, don't leak
                            _log.exception("console chat failed")
                            self._send(500, {"ok": False,
                                             "error": f"{type(exc).__name__}: {exc}"})
                            return
                        self._send(200, out)
                    else:
                        self._send(404, {"error": "not found"})
                except Exception as exc:  # noqa: BLE001 - never leak a traceback
                    _log.exception("console api error on %s %s", method, path)
                    self._send(500, {"error": type(exc).__name__})

            def do_GET(self) -> None:  # noqa: N802
                self._handle("GET")

            def do_POST(self) -> None:  # noqa: N802
                self._handle("POST")

        return Handler

    def serve(self, host: str = "127.0.0.1", port: int = 8788) -> int:
        httpd = ThreadingHTTPServer((host, port), self.handler_class())
        httpd.daemon_threads = True
        _log.info("web console on http://%s:%s (key %s…)", host, port, self.token[:4])
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:  # pragma: no cover  # noqa: E103 - deliberate top-level shutdown hook
            pass
        finally:
            httpd.shutdown()
            httpd.server_close()
            self.stop()
        return 0


# ── the page ─────────────────────────────────────────────────────────────────


def console_page() -> str:
    """The single-page console. No external assets — runs offline.

    Auth flow: the key lives in the URL fragment (``#key=…``) or is typed
    on the lock screen; it is kept in sessionStorage and sent as a
    Bearer header on every /api call.  The page itself carries no data.
    """
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nm console</title>
<style>
  :root {
    --bg: #0b0e14; --panel: #11151f; --panel2: #161c29;
    --line: #232b3d; --text: #d7dee9; --dim: #7d8aa0;
    --accent: #7ee787; --accent2: #58a6ff; --user: #1f2937;
    --danger: #f85149;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    background:
      radial-gradient(1200px 500px at 80% -10%, rgba(88,166,255,.08), transparent 60%),
      radial-gradient(900px 400px at 10% 110%, rgba(126,231,135,.06), transparent 60%),
      var(--bg);
    color: var(--text);
    font: 15px/1.5 "SF Mono", ui-monospace, Menlo, Consolas, monospace;
    display: flex; flex-direction: column; height: 100dvh;
  }
  header {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px; border-bottom: 1px solid var(--line);
    background: rgba(17,21,31,.85); backdrop-filter: blur(6px);
  }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent);
         box-shadow: 0 0 8px var(--accent); }
  .dot.busy { background: var(--accent2); box-shadow: 0 0 8px var(--accent2); }
  h1 { font-size: 14px; font-weight: 600; letter-spacing: .08em; }
  h1 span { color: var(--dim); font-weight: 400; }
  .mood { margin-left: auto; color: var(--dim); font-size: 12px; }
  #log { flex: 1; overflow-y: auto; padding: 18px 16px 8px;
         display: flex; flex-direction: column; gap: 10px; }
  .msg { max-width: 78%; padding: 9px 13px; border-radius: 12px;
         white-space: pre-wrap; word-wrap: break-word; animation: pop .15s ease-out; }
  @keyframes pop { from { opacity: 0; transform: translateY(4px);} to { opacity: 1; } }
  .msg.user { align-self: flex-end; background: var(--user);
              border: 1px solid #2c3a4f; border-bottom-right-radius: 4px; }
  .msg.her  { align-self: flex-start; background: var(--panel2);
              border: 1px solid var(--line); border-bottom-left-radius: 4px; }
  .msg.her .tag { display: block; font-size: 10px; color: var(--accent);
                  letter-spacing: .12em; margin-bottom: 4px; }
  .msg.sys  { align-self: center; color: var(--dim); font-size: 12px;
              background: none; padding: 2px; }
  .typing { align-self: flex-start; color: var(--dim); font-size: 12px;
            padding: 6px 13px; }
  .typing i { display: inline-block; width: 5px; height: 5px; margin-right: 3px;
              border-radius: 50%; background: var(--dim); animation: blink 1.2s infinite; }
  .typing i:nth-child(2) { animation-delay: .2s; }
  .typing i:nth-child(3) { animation-delay: .4s; }
  @keyframes blink { 0%,80%,100% { opacity: .25; } 40% { opacity: 1; } }
  footer { padding: 12px 16px 14px; border-top: 1px solid var(--line);
           background: rgba(17,21,31,.85); }
  .row { display: flex; gap: 8px; }
  textarea {
    flex: 1; resize: none; height: 44px; max-height: 140px;
    background: var(--panel); color: var(--text);
    border: 1px solid var(--line); border-radius: 10px; padding: 11px 13px;
    font: inherit; outline: none;
  }
  textarea:focus { border-color: var(--accent2); }
  button {
    background: var(--accent); color: #06130a; border: 0; border-radius: 10px;
    padding: 0 20px; font: inherit; font-weight: 700; cursor: pointer;
  }
  button:disabled { opacity: .4; cursor: default; }
  /* lock screen */
  #lock { position: fixed; inset: 0; background: var(--bg);
          display: flex; align-items: center; justify-content: center; z-index: 10; }
  .card { width: min(420px, 90vw); background: var(--panel);
          border: 1px solid var(--line); border-radius: 16px; padding: 34px 30px;
          text-align: center; }
  .card h2 { font-size: 16px; letter-spacing: .1em; margin-bottom: 6px; }
  .card p { color: var(--dim); font-size: 12.5px; margin-bottom: 20px; }
  .card input {
    width: 100%; background: var(--panel2); color: var(--text);
    border: 1px solid var(--line); border-radius: 10px; padding: 12px;
    font: inherit; letter-spacing: .06em; outline: none; text-align: center;
  }
  .card input:focus { border-color: var(--accent2); }
  .card button { width: 100%; margin-top: 14px; padding: 12px; }
  .err { color: var(--danger); font-size: 12px; margin-top: 10px; min-height: 16px; }
</style>
</head>
<body>
  <div id="lock">
    <div class="card">
      <h2>NM CONSOLE</h2>
      <p>enter the access key from the terminal
         (the one printed when <b>nm web</b> started)</p>
      <input id="key" type="password" autocomplete="off"
             placeholder="access key" spellcheck="false">
      <div class="err" id="lockerr"></div>
      <button id="unlock">unlock</button>
    </div>
  </div>

  <header>
    <div class="dot" id="dot"></div>
    <h1>NM <span>console</span></h1>
    <div class="mood" id="mood">…</div>
  </header>
  <div id="log"></div>
  <footer>
    <div class="row">
      <textarea id="input" placeholder="talk to her…  (slash commands work: /help)"
                rows="1"></textarea>
      <button id="send">send</button>
    </div>
  </footer>

<script>
(function () {
  "use strict";
  var KEY = "";
  var log = document.getElementById("log");
  var input = document.getElementById("input");
  var sendBtn = document.getElementById("send");
  var moodEl = document.getElementById("mood");
  var dot = document.getElementById("dot");
  var lock = document.getElementById("lock");
  var lockErr = document.getElementById("lockerr");
  var typingEl = null;
  var busy = false;

  function key() { return KEY; }

  function api(path, opts) {
    opts = opts || {};
    opts.headers = opts.headers || {};
    if (KEY) opts.headers["Authorization"] = "Bearer " + KEY;
    return fetch(path, opts).then(function (r) {
      if (r.status === 401) { throw new Error("unauthorized"); }
      return r.json().then(function (j) {
        if (!r.ok) { throw new Error(j.error || ("http " + r.status)); }
        return j;
      });
    });
  }

  function addMsg(cls, text, tag) {
    var div = document.createElement("div");
    div.className = "msg " + cls;
    if (tag) {
      var t = document.createElement("span");
      t.className = "tag"; t.textContent = tag;
      div.appendChild(t);
    }
    div.appendChild(document.createTextNode(text));
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    return div;
  }

  function showTyping(on) {
    if (on && !typingEl) {
      typingEl = document.createElement("div");
      typingEl.className = "typing";
      typingEl.innerHTML = "<i></i><i></i><i></i>";
      log.appendChild(typingEl);
      log.scrollTop = log.scrollHeight;
    } else if (!on && typingEl) {
      typingEl.remove();
      typingEl = null;
    }
  }

  function unlock(k) {
    KEY = (k || "").trim();
    if (!KEY) { lockErr.textContent = "key is empty"; return; }
    lockErr.textContent = "";
    api("/api/status").then(function (s) {
      lock.style.display = "none";
      try { sessionStorage.setItem("nm_key", KEY); } catch (e) {}
      addMsg("sys", "unlocked — " + s.version +
        (s.mood ? "  ·  mood: " + s.mood : ""));
      refreshMood();
    }).catch(function (e) {
      lockErr.textContent = "wrong key — " + e.message;
    });
  }

  function refreshMood() {
    api("/api/status").then(function (s) {
      moodEl.textContent = (s.mood || "idle") +
        (s.busy ? "  ·  thinking" : "");
      dot.className = "dot" + (s.busy ? " busy" : "");
    }).catch(function () {});
  }

  function send() {
    var text = input.value.trim();
    if (!text || busy) { return; }
    busy = true;
    sendBtn.disabled = true;
    input.value = "";
    input.style.height = "44px";
    addMsg("user", text);
    showTyping(true);
    api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    }).then(function (out) {
      showTyping(false);
      var parts = out.parts && out.parts.length ? out.parts : [out.reply];
      parts.forEach(function (p) {
        addMsg("her", p, out.mood ? ("mood: " + out.mood) : null);
      });
      if (!parts.length) { addMsg("sys", "(no reply)"); }
    }).catch(function (e) {
      showTyping(false);
      if (e.message === "unauthorized") {
        location.hash = "";
        lock.style.display = "flex";
        lockErr.textContent = "key rejected — re-enter it";
      } else {
        addMsg("sys", "error: " + e.message);
      }
    }).then(function () {
      busy = false;
      sendBtn.disabled = false;
      input.focus();
      refreshMood();
    });
  }

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  input.addEventListener("input", function () {
    input.style.height = "44px";
    input.style.height = Math.min(140, input.scrollHeight) + "px";
  });

  // auto-unlock from #key=… fragment (the URL the terminal prints)
  var m = location.hash.match(/key=([A-Za-z0-9._~\\-]+)/);
  if (m) {
    document.getElementById("key").value = decodeURIComponent(m[1]);
    unlock(m[1]);
  }
  var saved = null;
  try { saved = sessionStorage.getItem("nm_key"); } catch (e) {}
  if (saved && !m) { unlock(saved); }
  document.getElementById("unlock").addEventListener("click", function () {
    unlock(document.getElementById("key").value);
  });
  document.getElementById("key").addEventListener("keydown", function (e) {
    if (e.key === "Enter") { unlock(e.target.value); }
  });
  setInterval(refreshMood, 4000);
})();
</script>
</body>
</html>
"""
