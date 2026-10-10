#!/usr/bin/env node
/**
 * whatsapp-bridge.mjs — WhatsApp Web session holder for Devon's WhatsAppAdapter.
 *
 * What it does:
 *   1. Logs into WhatsApp via Baileys (QR code on first run, session saved to .creds/).
 *   2. Listens on 127.0.0.1:8787 and speaks a JSON-lines protocol with the Python
 *      adapter (nomorals/social/chat/whatsapp.py).
 *
 * Protocol (one JSON object per line):
 *
 *   bridge -> python
 *     {"type":"status","state":"open|closed","user":"<jid>"}
 *     {"type":"qr","data":"<qr string>"}
 *     {"type":"pairing_code","data":"<8-char code>"}   (when WA_PAIRING_CODE=1)
 *     {"type":"message","chat":{"id","kind","title"},"from":{"id","name"},
 *      "text":"...","media":[{"path","mime","kind"}],"reply_to":"",
 *      "mentioned":false,"ts":<ms>}
 *     {"type":"receipt","chat":"<jid>","ids":["<msg>",...],"kind":"delivered|read"}
 *     {"id":"<req>","ok":true|false,"error":"", ...extra}
 *
 *   python -> bridge
 *     {"id":"<req>","cmd":"send","chat":"<jid>","text":"...","reply_to":""}
 *     {"id":"<req>","cmd":"send_media","chat":"<jid>","path":"...","caption":"...","ptt":false}
 *     {"id":"<req>","cmd":"typing","chat":"<jid>","seconds":3,
 *      "presence":"composing|recording"}
 *     {"id":"<req>","cmd":"history","chat":"<jid>","limit":20}
 *     {"id":"<req>","cmd":"read","chat":"<jid>"}
 *     {"id":"<req>","cmd":"chats","limit":30}
 *     {"id":"<req>","cmd":"status"}
 *
 * Run:  node whatsapp-bridge.mjs [--port 8787] [--creds ./.creds] [--media ./media]
 * First run prints a QR code in the terminal — scan it with WhatsApp
 * (Settings -> Linked Devices -> Link a Device). The session persists in
 * .creds/ afterwards; guard that directory like a password.
 * Reconnects use a reason-classified policy (see RECONNECT POLICY below) with
 * jittered exponential backoff (5s -> 120s cap, retry-budget ceiling, then
 * park + notify). Terminal codes (401 logged-out, 500 bad-session, 440
 * replaced, 411 md-mismatch, 403 forbidden) stop retries and notify the
 * owner instead of retrying blindly.
 * Reconnect state persists in <creds>/reconnect-state.json so a process
 * restart does not reset the attempt ladder.
 * Alternatively set WA_PAIRING_CODE=1 + WA_PAIRING_NUMBER=<int'l digits,
 * no +> for an 8-char code you type into the phone instead of scanning.
 */

import net from 'node:net';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import qrcode from 'qrcode-terminal';
import makeWASocket, {
  useMultiFileAuthState,
  DisconnectReason,
  downloadMediaMessage,
  fetchLatestBaileysVersion,
  jidNormalizedUser,
} from '@whiskeysockets/baileys';
import registerGroupCommands from './whatsapp-groups.mjs';
import registerGroupAdminCommands from './whatsapp-groups-admin.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};
const PORT = parseInt(opt('--port', process.env.WA_BRIDGE_PORT || '8787'), 10);
const CREDS_DIR = path.resolve(here, opt('--creds', process.env.WA_CREDS_DIR || './.creds'));
const MEDIA_DIR = path.resolve(here, opt('--media', process.env.WA_MEDIA_DIR || './media'));
// Pairing-code login (no QR scan): set WA_PAIRING_CODE=1 and WA_PAIRING_NUMBER
// to the phone number (international format, no "+") to get an 8-char code
// printed here that you type into WhatsApp > Linked Devices > Link with code.
const PAIRING_CODE = process.env.WA_PAIRING_CODE === '1' || process.env.WA_PAIRING_CODE === 'true';
const PAIRING_NUMBER = (process.env.WA_PAIRING_NUMBER || '').replace(/[^0-9]/g, '');
fs.mkdirSync(MEDIA_DIR, { recursive: true });

const log = (...a) => console.log(new Date().toISOString(), '[bridge]', ...a);

// ── RECONNECT POLICY ────────────────────────────────────────────────────────
// Reason-classified, data-driven (strategy table — not if/else spaghetti).
// Mined from Baileys 6.7.24's actual enum:
//   bridge/node_modules/@whiskeysockets/baileys/lib/Types/index.d.ts (line 25)
//   bridge/node_modules/@whiskeysockets/baileys/lib/Types/index.js  (lines 14-24)
//   connection.update payload: lib/Socket/socket.js lines 252-262 —
//     { connection, lastDisconnect: { error: Boom, date }, qr, ... }
//     where error?.output?.statusCode is the DisconnectReason code.
//   Baileys' own keep-alive: socket.js lines 283-315 — kills a silent
//     connection after keepAliveIntervalMs(30s)+5s with code 408.
//   ws liveness: wa.ws is a WebSocketClient exposing isOpen/isClosed/isClosing
//     (lib/Socket/Client/websocket.js + .d.ts) and close().
// Semantics:
//   401 loggedOut / 500 badSession          -> terminal: session dead, re-auth required
//   440 connectionReplaced                 -> terminal: another client took the
//                                             session over; auto-reconnect would
//                                             fight it, so stop and notify
//   411 multideviceMismatch / 403 forbidden-> terminal: server refuses this client
//   428 / 408 (closed/lost/timedOut) / 515 / 503 -> retry with backoff
//   unknown / missing code                 -> conservative retry, same budget cap
const RECONNECT_BASE_S = 5;
const RECONNECT_CAP_S = 120;
const RECONNECT_MAX_ATTEMPTS = 15;   // then park and notify the owner
const RECONNECT_STATE_FILE = path.join(CREDS_DIR, 'reconnect-state.json');
const RECONNECT_RESET_WINDOW_MS = 15 * 60 * 1000; // old outages don't keep a stale ladder
const HEARTBEAT_MS = 30_000;

const ACTION = { RETRY: 'retry', STOP: 'stop' };
const POLICY_TABLE = [
  { codes: [DisconnectReason.loggedOut, DisconnectReason.badSession],
    action: ACTION.STOP, reauth: true,
    label: (c) => c === DisconnectReason.loggedOut ? 'logged-out' : 'bad-session',
    detail: 'session invalidated — delete .creds/ and re-scan/re-pair' },
  { codes: [DisconnectReason.connectionReplaced],
    action: ACTION.STOP, reauth: false,
    label: () => 'connection-replaced',
    detail: 'another client took this session over — not auto-reconnecting' },
  { codes: [DisconnectReason.multideviceMismatch, DisconnectReason.forbidden],
    action: ACTION.STOP, reauth: false,
    label: (c) => c === DisconnectReason.multideviceMismatch ? 'multidevice-mismatch' : 'forbidden',
    detail: 'server refuses this client — owner action required' },
  { codes: [DisconnectReason.connectionClosed, DisconnectReason.connectionLost,
            DisconnectReason.timedOut, DisconnectReason.restartRequired,
            DisconnectReason.unavailableService],
    action: ACTION.RETRY, reauth: false,
    label: (c) => DisconnectReason[c]?.replace(/([A-Z])/g, '-$1').toLowerCase() || `code-${c}`,
    detail: 'transient — backing off and retrying' },
];
function classifyDisconnect(code) {
  for (const entry of POLICY_TABLE) {
    if (entry.codes.includes(code)) {
      return { action: entry.action, reauth: entry.reauth,
               label: entry.label(code), detail: entry.detail };
    }
  }
  return { action: ACTION.RETRY, reauth: false,
           label: code == null ? 'unknown-no-code' : `unknown-code-${code}`,
           detail: 'unrecognized code — conservative retry under the attempt budget' };
}

// ── reconnect state (persisted so a restart keeps the ladder/context) ────────
let reconnectAttempt = 0;
let lastDisconnectCode = null;
let lastDisconnectLabel = null;
let lastDisconnectAt = 0;
let terminalReason = null;   // set when we stop retrying (owner must act)
let nextRetryAt = 0;         // ms epoch, 0 when no retry scheduled
let reconnectTimer = null;
let connecting = false;
let connState = 'closed';    // mirror of the last connection.update connection value
let lastOpenAt = 0;

function saveReconnectState() {
  try {
    fs.writeFileSync(RECONNECT_STATE_FILE, JSON.stringify({
      attempt: reconnectAttempt,
      lastDisconnectCode,
      lastDisconnectLabel,
      lastDisconnectAt,
      terminalReason,
      savedAt: Date.now(),
    }, null, 2));
  } catch (e) { log('reconnect-state save failed:', e.message); }
}
function loadReconnectState() {
  try {
    const raw = fs.readFileSync(RECONNECT_STATE_FILE, 'utf8');
    const s = JSON.parse(raw);
    const age = Date.now() - (s.lastDisconnectAt || 0);
    if (age > RECONNECT_RESET_WINDOW_MS) {
      log('reconnect-state: last outage is old, starting a fresh attempt ladder');
      return;
    }
    reconnectAttempt = Math.max(0, parseInt(s.attempt || '0', 10));
    lastDisconnectCode = s.lastDisconnectCode ?? null;
    lastDisconnectLabel = s.lastDisconnectLabel || null;
    lastDisconnectAt = s.lastDisconnectAt || 0;
    terminalReason = s.terminalReason || null;
    if (reconnectAttempt > 0 || terminalReason) {
      log(`reconnect-state restored: attempt=${reconnectAttempt} last=${lastDisconnectLabel} terminal=${terminalReason || 'no'}`);
    }
  } catch { /* no state file yet — fresh start */ }
}

function reconnectDelayS() {
  reconnectAttempt += 1;
  const exp = Math.min(RECONNECT_BASE_S * 2 ** (reconnectAttempt - 1), RECONNECT_CAP_S);
  return exp * (0.7 + Math.random() * 0.6); // ±~30% jitter
}

// Rich status broadcast. Additive fields only — the Python adapter
// (nomorals/social/chat/whatsapp.py, kind == "status") reads only
// "user"/"state" and ignores the rest, so this stays backward compatible.
function statusPayload(state) {
  const nextInS = nextRetryAt > Date.now() ? Math.round((nextRetryAt - Date.now()) / 1000) : 0;
  return {
    type: 'status',
    state,                 // open | closed | reconnecting | parked
    user: meJid,
    reconnect_attempt: reconnectAttempt,
    next_retry_in_s: nextInS,
    last_disconnect_code: lastDisconnectCode,
    last_disconnect_label: lastDisconnectLabel,
    terminal_reason: terminalReason,
  };
}
function broadcastStatus(state, extra = {}) {
  broadcast({ ...statusPayload(state), ...extra });
}

function park(reason, detail) {
  terminalReason = reason;
  nextRetryAt = 0;
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  connState = 'parked';
  saveReconnectState();
  log(`PARKED: ${reason}${detail ? ` — ${detail}` : ''} (owner action required)`);
  broadcastStatus('parked');
}

// Rejection-safe connect: every failure is caught, logged, persisted, and
// routed through the policy — never an unhandled rejection.
async function safeConnect() {
  if (connecting) { log('connect already in progress — skipping'); return; }
  if (terminalReason) { log('parked, not reconnecting:', terminalReason); return; }
  connecting = true;
  try {
    await connect();
  } catch (e) {
    log('connect failed:', e?.message || e);
    // Treat a hard connect failure like a transient close and schedule again.
    onConnectionClosed(null, e);
  } finally {
    connecting = false;
  }
}

function scheduleReconnect() {
  if (terminalReason) return;
  if (reconnectAttempt >= RECONNECT_MAX_ATTEMPTS) {
    park('retry-budget-exhausted',
      `stopped after ${reconnectAttempt} attempts — inspect .creds/ and network, then restart the bridge`);
    return;
  }
  const delay = reconnectDelayS();
  nextRetryAt = Date.now() + delay * 1000;
  log(`reconnecting in ${delay.toFixed(0)}s (attempt ${reconnectAttempt}/${RECONNECT_MAX_ATTEMPTS})`);
  saveReconnectState();
  broadcastStatus('reconnecting');
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    safeConnect().catch((e) => {
      // belt and braces: safeConnect already catches; this is the last net
      log('unexpected reconnect rejection:', e?.message || e);
      scheduleReconnect();
    });
  }, delay * 1000);
}

// Single entry point for every close/failure: classify via the policy table,
// broadcast rich status, then retry | stop-and-notify.
function onConnectionClosed(code, rawError) {
  connState = 'close';
  const { action, reauth, label, detail } = classifyDisconnect(code);
  lastDisconnectCode = code ?? null;
  lastDisconnectLabel = label;
  lastDisconnectAt = Date.now();
  log(`connection closed (code ${code ?? 'n/a'} / ${label})${rawError && code == null ? `: ${rawError?.message || rawError}` : ''}`);
  if (action === ACTION.STOP) {
    broadcastStatus('closed');
    park(label + (reauth ? ' — re-auth required' : ''), detail);
    return;
  }
  broadcastStatus('closed');
  scheduleReconnect();
}

// Heartbeat: catches half-open sockets — we believe we're open but the
// underlying ws is gone and Baileys never emitted `close` (its own 30s+5s
// keep-alive covers server silence; this covers the client-side blind spot).
function heartbeat() {
  try {
    if (connState !== 'open') return;
    const wsOpen = wa?.ws?.isOpen === true;
    if (!wa || !wsOpen) {
      log(`heartbeat: state=open but ws ${!wa ? 'missing' : 'not open'} — forcing reconnect cycle`);
      connState = 'close';
      try { wa?.ws?.close?.()?.catch?.(() => {}); } catch { /* ignore */ }
      // Run the policy even if Baileys' internal close never fires.
      onConnectionClosed(DisconnectReason.connectionLost, new Error('heartbeat: half-open socket'));
    }
  } catch (e) {
    log('heartbeat error:', e?.message || e);
  }
}

// ── TCP plumbing ─────────────────────────────────────────────────────────────
const clients = new Set();
function broadcast(obj) {
  const line = JSON.stringify(obj) + '\n';
  for (const s of clients) {
    try { s.write(line); } catch { /* drop dead client */ }
  }
}
function reply(sock, id, obj) {
  try { sock.write(JSON.stringify({ id, ...obj }) + '\n'); } catch { /* ignore */ }
}

// ── message memory (ring buffer per chat; backs `history` + `chats`) ─────────
const MAX_KEEP = 200;
const recentByChat = new Map(); // jid -> [msg, ...] newest last
const chatTitles = new Map();   // jid -> title
const msgCache = new Map();     // stanzaId -> raw msg (for quoted replies)
function remember(jid, msg) {
  let arr = recentByChat.get(jid);
  if (!arr) { arr = []; recentByChat.set(jid, arr); }
  arr.push(msg);
  if (arr.length > MAX_KEEP) arr.splice(0, arr.length - MAX_KEEP);
  if (msgCache.size > 2000) {
    for (const k of msgCache.keys()) { msgCache.delete(k); if (msgCache.size < 1500) break; }
  }
}

// ── Baileys helpers ──────────────────────────────────────────────────────────
const kindOf = (jid) => (jid.endsWith('@g.us') ? 'group' : 'dm');

function textOf(m) {
  const msg = m.message || {};
  return (
    msg.conversation ||
    msg.extendedTextMessage?.text ||
    msg.imageMessage?.caption ||
    msg.videoMessage?.caption ||
    ''
  );
}

function replyToOf(m) {
  const ctx = m.message?.extendedTextMessage?.contextInfo
    || m.message?.imageMessage?.contextInfo
    || m.message?.videoMessage?.contextInfo;
  return ctx?.stanzaId || '';
}

function mentionedJidOf(m) {
  const ctx = m.message?.extendedTextMessage?.contextInfo
    || m.message?.imageMessage?.contextInfo
    || m.message?.videoMessage?.contextInfo
    || m.message?.audioMessage?.contextInfo;
  const jids = ctx?.mentionedJid || [];
  return Array.isArray(jids) ? jids : [];
}

function mediaKindOf(m) {
  const msg = m.message || {};
  if (msg.imageMessage) return 'image';
  if (msg.videoMessage) return 'video';
  if (msg.audioMessage) return msg.audioMessage.ptt ? 'voice' : 'audio';
  if (msg.documentMessage) return 'file';
  if (msg.stickerMessage) return 'sticker';
  return '';
}

const EXT_MIME = {
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
  '.webp': 'image/webp', '.gif': 'image/gif', '.mp4': 'video/mp4',
  '.mov': 'video/quicktime', '.mp3': 'audio/mpeg', '.ogg': 'audio/ogg',
  '.oga': 'audio/ogg', '.wav': 'audio/wav', '.m4a': 'audio/mp4',
  '.pdf': 'application/pdf',
};

async function downloadInbound(m) {
  // Returns {path, mime, kind} or null.
  try {
    const kind = mediaKindOf(m);
    if (!kind) return null;
    const buf = await downloadMediaMessage(m, 'buffer', {});
    const inner = m.message?.imageMessage || m.message?.videoMessage
      || m.message?.audioMessage || m.message?.documentMessage
      || m.message?.stickerMessage || {};
    const mime = inner.mimetype || 'application/octet-stream';
    const ext = inner.fileName?.includes('.')
      ? '.' + inner.fileName.split('.').pop().toLowerCase()
      : ({ image: '.jpg', video: '.mp4', voice: '.ogg', audio: '.ogg', sticker: '.webp' }[kind] || '.bin');
    const name = `${Date.now()}-${(m.key.id || 'm').replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 12)}${ext}`;
    const p = path.join(MEDIA_DIR, name);
    fs.writeFileSync(p, buf);
    return { path: p, mime, kind };
  } catch (e) {
    log('media download failed:', e.message);
    return null;
  }
}

// ── WhatsApp connection ──────────────────────────────────────────────────────
let wa = null;
let meJid = '';

async function connect() {
  const { state, saveCreds } = await useMultiFileAuthState(CREDS_DIR);
  const { version } = await fetchLatestBaileysVersion().catch(() => ({ version: undefined }));
  wa = makeWASocket({
    auth: state,
    printQRInTerminal: false,
    ...(version ? { version } : {}),
  });
  wa.ev.on('creds.update', saveCreds);

  wa.ev.on('connection.update', (u) => {
    const { connection, lastDisconnect, qr } = u;
    if (qr) {
      if (PAIRING_CODE && PAIRING_NUMBER) {
        // Pairing-code login: an 8-char code the owner types into the phone.
        wa.requestPairingCode(PAIRING_NUMBER).then(
          (code) => {
            log(`PAIRING CODE for +${PAIRING_NUMBER}: ${code} (type into WhatsApp > Linked Devices)`);
            broadcast({ type: 'pairing_code', data: code });
          },
          (e) => log('pairing code request failed:', e?.message || e),
        );
      } else {
        log('QR received — scan with WhatsApp (Linked Devices).');
        try { qrcode.generate(qr, { small: true }); } catch { /* ignore */ }
        broadcast({ type: 'qr', data: qr });
      }
    }
    if (connection === 'open') {
      meJid = jidNormalizedUser(wa.user?.id || '');
      connState = 'open';
      lastOpenAt = Date.now();
      if (terminalReason) {
        // A previous park is cleared by a genuinely successful (re)connect —
        // e.g. after the owner re-scanned and restarted.
        log('connected after park — clearing terminal state:', terminalReason);
      }
      terminalReason = null;
      reconnectAttempt = 0; // healthy connection resets the backoff ladder
      nextRetryAt = 0;
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
      saveReconnectState();
      log('connected as', meJid);
      broadcastStatus('open');
    }
    if (connection === 'close') {
      const code = lastDisconnect?.error?.output?.statusCode;
      onConnectionClosed(code, lastDisconnect?.error);
    }
  });

  // Delivery receipts: sent → delivered → read, per message stanza.
  // Only broadcast; the Python adapter tracks them per message id.
  wa.ev.on('messages.update', (updates) => {
    try {
      const byChat = new Map();
      for (const u of updates || []) {
        const status = u?.update?.status;
        if (status === undefined || status === null) continue;
        const kind = status >= 4 ? 'read' : status >= 3 ? 'delivered' : '';
        if (!kind) continue;
        const jid = jidNormalizedUser(u?.key?.remoteJid || '');
        if (!jid) continue;
        let bucket = byChat.get(jid);
        if (!bucket) { bucket = { delivered: new Set(), read: new Set() }; byChat.set(jid, bucket); }
        const id = u.key?.id || '';
        if (id) bucket[kind].add(id);
      }
      for (const [jid, bucket] of byChat) {
        for (const kind of ['delivered', 'read']) {
          const ids = [...bucket[kind]];
          if (ids.length) broadcast({ type: 'receipt', chat: jid, ids, kind });
        }
      }
    } catch (e) {
      log('receipt handler error:', e?.message || e);
    }
  });

  wa.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify') return;
    for (const m of messages) {
      try {
        if (!m.message) continue;
        const jid = jidNormalizedUser(m.key.remoteJid || '');
        if (!jid || jid === 'status@broadcast') continue;
        // Own outgoing messages (shared-account mode): forward as
        // informational own_message events so the bot can track the
        // owner's replies (double-reply discipline, per-chat profile).
        // Never processed as inbound — the brain stays silent on them.
        if (m.key.fromMe) {
          const ownOut = {
            type: 'own_message',
            chat: { id: jid, kind: kindOf(jid), title: chatTitles.get(jid) || '' },
            text: textOf(m),
            ts: Number(m.messageTimestamp || Date.now() / 1000) * 1000,
          };
          remember(jid, { ...ownOut, incoming: false, sender: '', key: m.key });
          broadcast(ownOut);
          continue;
        }
        const senderId = m.key.participant ? jidNormalizedUser(m.key.participant) : jid;
        const kind = kindOf(jid);
        let title = chatTitles.get(jid) || '';
        if (kind === 'group' && !title) {
          try {
            const meta = await wa.groupMetadata(jid);
            title = meta.subject || '';
            chatTitles.set(jid, title);
          } catch { /* ignore */ }
        }
        if (!title) title = m.pushName || senderId.split('@')[0];
        else if (kind === 'dm') chatTitles.set(jid, m.pushName || title);
        const media = await downloadInbound(m);
        // @-mention of this account (groups): drives the brain's
        // should-speak-in-group exactly like Telegram mentions.
        const mentioned = mentionedJidOf(m).some(
          (j) => jidNormalizedUser(String(j)) === meJid);
        const out = {
          type: 'message',
          chat: { id: jid, kind, title },
          from: { id: senderId, name: m.pushName || '' },
          text: textOf(m),
          media: media ? [media] : [],
          reply_to: replyToOf(m),
          mentioned,
          ts: Number(m.messageTimestamp || Date.now() / 1000) * 1000,
        };
        if (m.key.id) msgCache.set(m.key.id, m);
        remember(jid, { ...out, incoming: true, sender: m.pushName || '', key: m.key });
        broadcast(out);
      } catch (e) {
        log('upsert handler error:', e.message);
      }
    }
  });
}

// ── command handlers ─────────────────────────────────────────────────────────
function extOf(p) {
  const i = p.lastIndexOf('.');
  return i >= 0 ? p.slice(i).toLowerCase() : '';
}

async function cmdSend({ chat, text, reply_to }) {
  const content = { text: text || '' };
  const opts = {};
  if (reply_to && msgCache.has(reply_to)) {
    opts.quoted = msgCache.get(reply_to); // raw WAMessage -> quoted reply
  }
  return wa.sendMessage(chat, content, opts);
}

async function cmdSendMedia({ chat, path: p, caption, ptt }) {
  if (!p || !fs.existsSync(p)) throw new Error(`media file not found: ${p}`);
  const ext = extOf(p);
  const mime = EXT_MIME[ext];
  const name = path.basename(p);
  const cap = caption || '';
  if (['.jpg', '.jpeg', '.png', '.webp'].includes(ext)) {
    return wa.sendMessage(chat, { image: { url: p }, caption: cap, mimetype: mime });
  }
  if (['.mp4', '.mov', '.gif'].includes(ext)) {
    return wa.sendMessage(chat, { video: { url: p }, caption: cap, mimetype: mime });
  }
  if (['.mp3', '.ogg', '.oga', '.wav', '.m4a'].includes(ext)) {
    return wa.sendMessage(chat, { audio: { url: p }, mimetype: mime || 'audio/ogg', ptt: !!ptt });
  }
  return wa.sendMessage(chat, { document: { url: p }, mimetype: mime || 'application/octet-stream', fileName: name, caption: cap });
}

async function handleCommand(sock, req) {
  const { id, cmd } = req;
  if (!wa) return reply(sock, id, { ok: false, error: 'whatsapp not connected yet' });
  try {
    switch (cmd) {
      case 'send': {
        await cmdSend(req);
        return reply(sock, id, { ok: true });
      }
      case 'send_media': {
        await cmdSendMedia(req);
        return reply(sock, id, { ok: true });
      }
      case 'typing': {
        const secs = Math.max(1, Math.min(30, parseInt(req.seconds || '3', 10)));
        // "recording" presence for voice-note replies, else regular typing.
        const presence = req.presence === 'recording' ? 'recording' : 'composing';
        await wa.sendPresenceUpdate(presence, req.chat);
        setTimeout(() => wa?.sendPresenceUpdate('paused', req.chat).catch(() => {}), secs * 1000);
        return reply(sock, id, { ok: true });
      }
      case 'history': {
        const arr = recentByChat.get(jidNormalizedUser(req.chat || '')) || [];
        const limit = Math.max(1, parseInt(req.limit || '20', 10));
        const msgs = arr.slice(-limit).map((m) => ({
          incoming: m.incoming !== false,
          text: m.text || '',
          sender: m.sender || (m.from && m.from.name) || '',
          media: m.media || [],
          reply_to: m.reply_to || '',
          ts: m.ts || Date.now(),
        }));
        return reply(sock, id, { ok: true, messages: msgs });
      }
      case 'read': {
        const jid = jidNormalizedUser(req.chat || '');
        const arr = recentByChat.get(jid) || [];
        const last = [...arr].reverse().find((m) => m.incoming !== false && m.key);
        // best-effort: receipts need the raw key; the ack is what the adapter checks
        if (last) await wa.readMessages([last.key]).catch(() => {});
        return reply(sock, id, { ok: true });
      }
      case 'chats': {
        const limit = Math.max(1, parseInt(req.limit || '30', 10));
        const list = [...recentByChat.entries()]
          .map(([jid2, arr]) => ({
            id: jid2,
            title: chatTitles.get(jid2) || jid2.split('@')[0],
            kind: kindOf(jid2),
            ts: (arr[arr.length - 1] || {}).ts || 0,
          }))
          .sort((a, b) => b.ts - a.ts)
          .slice(0, limit);
        return reply(sock, id, { ok: true, chats: list });
      }
      case 'status': {
        return reply(sock, id, { ok: true, user: meJid, state: wa ? 'open' : 'closed' });
      }
      default:
        // workstream 3: group/community commands (read-only) before "unknown cmd",
        // then the mutating admin commands (owner-confirmed on the Python side).
        return (await registerGroupCommands({ getSocket: () => wa, reply, log })(sock, req))
          || (await registerGroupAdminCommands({ getSocket: () => wa, reply, log })(sock, req))
          || reply(sock, id, { ok: false, error: `unknown cmd: ${cmd}` });
    }
  } catch (e) {
    return reply(sock, id, { ok: false, error: e.message || String(e) });
  }
}

// ── TCP server ───────────────────────────────────────────────────────────────
const server = net.createServer((sock) => {
  clients.add(sock);
  // send current state immediately so a fresh client knows where we stand
  sock.write(JSON.stringify(statusPayload(connState === 'open' ? 'open' : connState === 'parked' ? 'parked' : 'closed')) + '\n');
  let buf = '';
  sock.on('data', (chunk) => {
    buf += chunk.toString('utf8');
    let idx;
    while ((idx = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      let req;
      try { req = JSON.parse(line); } catch { continue; }
      if (req && req.cmd) handleCommand(sock, req);
    }
  });
  const drop = () => clients.delete(sock);
  sock.on('close', drop);
  sock.on('error', drop);
});

server.listen(PORT, '127.0.0.1', () => {
  log(`listening on 127.0.0.1:${PORT}; creds=${CREDS_DIR}; media=${MEDIA_DIR}`);
  loadReconnectState();
  setInterval(heartbeat, HEARTBEAT_MS);
  safeConnect();
});

process.on('SIGINT', () => process.exit(0));
process.on('SIGTERM', () => process.exit(0));
