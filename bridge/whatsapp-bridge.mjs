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
 *     {"type":"status","state":"open|closed","user":"<jid>","lid":"<lid>@lid"}
 *     {"type":"qr","data":"<qr string>"}
 *     {"type":"pairing_code","data":"<8-char code>"}   (when WA_PAIRING_CODE=1)
 *     {"type":"message","id":"<wa message id>","chat":{"id","kind","title"},
 *      "from":{"id","name"},"text":"...","media":[{"path","mime","kind"}],
 *      "reply_to":"","mentioned":false,"ts":<ms>}
 *     (a reply also carries "quoted_text" and "quoted_media":
 *      the quoted message's own text and media, downloaded at
 *      inbound — a reply quotes its target inside its own
 *      contextInfo, and those bytes are only reachable now.)
 *     {"type":"receipt","chat":"<jid>","ids":["<msg>",...],"kind":"delivered|read"}
 *     {"id":"<req>","ok":true|false,"error":"", ...extra}
 *      (send/send_media replies add "message_id": the real WA message id)
 *
 *   python -> bridge
 *     {"id":"<req>","cmd":"send","chat":"<jid>","text":"...","reply_to":"",
 *      "buttons":[[["label","command"],…],…] (optional; rendered natively
 *      up to WhatsApp's 3-button cap, the rest as text action lines)}
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
 * Pairing codes are SERVER-CONFIRMED before they are printed (see
 * requestPairingCodeConfirmed below): a code you see is a code
 * WhatsApp accepted at stage 1; a refusal prints as a refusal
 * instead of a code that never existed server-side.
 */

import net from 'node:net';
import fs from 'node:fs';
import path from 'node:path';
import { randomBytes } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import qrcode from 'qrcode-terminal';
import makeWASocket, {
  useMultiFileAuthState,
  DisconnectReason,
  downloadContentFromMessage,
  downloadMediaMessage,
  fetchLatestBaileysVersion,
  jidNormalizedUser,
  aesEncryptCTR,
  bytesToCrockford,
  derivePairingCodeKey,
  getPlatformId,
  jidEncode,
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
// Presence (§15 always-online): the last state the presence verb
// successfully asserted, re-asserted on every (re)connect —
// Baileys' assertion does not survive its socket, and a bridge
// that only asserts when told leaves the account dark after any
// flap until the Python beat happens to run. '' = never asked.
let lastPresenceState = '';

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
// (nomorals/social/chat/whatsapp.py, kind == "status") reads
// "user"/"state"/"lid" and ignores the rest, so this stays
// backward compatible.
function statusPayload(state) {
  const nextInS = nextRetryAt > Date.now() ? Math.round((nextRetryAt - Date.now()) / 1000) : 0;
  return {
    type: 'status',
    state,                 // open | closed | reconnecting | parked
    user: meJid,
    lid: meLid,            // "" when the account has no LID
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
const kindOf = (jid) => (jid.endsWith('@g.us') ? 'group'
  : jid.endsWith('@newsletter') ? 'channel' : 'dm');

function textOf(m) {
  const msg = m.message || {};
  // Button/list taps re-enter as the command they carry — the
  // same contract as Telegram callback taps ("/" + data), so a
  // keyboard authored once works on both platforms.
  const btnId = msg.buttonsResponseMessage?.selectedButtonId
    || msg.templateButtonReplyMessage?.selectedId
    || msg.listResponseMessage?.singleSelectReply?.selectedRowId
    || '';
  if (btnId) return `/${btnId}`;
  const flow = msg.interactiveResponseMessage
    ?.nativeFlowResponseMessage?.paramsJson;
  if (flow) {
    try {
      const parsed = JSON.parse(flow);
      if (parsed && parsed.id) return `/${parsed.id}`;
    } catch { /* not a button payload — fall through to text */ }
  }
  return (
    msg.conversation ||
    msg.extendedTextMessage?.text ||
    msg.imageMessage?.caption ||
    msg.videoMessage?.caption ||
    ''
  );
}

function contextInfoOf(m) {
  const msg = m.message || {};
  return msg.extendedTextMessage?.contextInfo
    || msg.imageMessage?.contextInfo
    || msg.videoMessage?.contextInfo
    || msg.audioMessage?.contextInfo
    || msg.documentMessage?.contextInfo
    || msg.stickerMessage?.contextInfo
    // Button/list taps: the response's context stanza names the
    // message that carried the buttons — the live-session
    // presentation path edits THAT message in place, so a tap
    // reports it as reply_to like a quote does (Go bridge parity).
    || msg.buttonsResponseMessage?.contextInfo
    || msg.templateButtonReplyMessage?.contextInfo
    || msg.listResponseMessage?.contextInfo
    || msg.interactiveResponseMessage?.contextInfo
    || null;
}

function replyToOf(m) {
  return contextInfoOf(m)?.stanzaId || '';
}

// The reply quote's payload: the quoted message rides INSIDE the
// reply's contextInfo.quotedMessage. The stanza id alone (all
// this bridge used to send) tells the Python side THAT a message
// was quoted, never WHAT — reply-media commands (/sticker on a
// replied photo) fell back to usage text while the photo sat in
// the quote. Download the quoted media now, the only moment its
// bytes are reachable, and carry the quoted text alongside.
async function quotedPayload(m) {
  const ctx = contextInfoOf(m);
  const quoted = ctx?.quotedMessage;
  if (!quoted) return {};
  const out = {};
  const qtext = textOf({ message: quoted });
  if (qtext) out.quoted_text = qtext;
  const kind = mediaKindOf({ message: quoted });
  if (kind) {
    const inner = quoted.imageMessage || quoted.videoMessage
      || quoted.audioMessage || quoted.documentMessage
      || quoted.stickerMessage || {};
    const mime = inner.mimetype || 'application/octet-stream';
    try {
      const dlType = { image: 'image', video: 'video', voice: 'audio',
        audio: 'audio', file: 'document', sticker: 'sticker' }[kind];
      const stream = await downloadContentFromMessage(inner, dlType);
      const chunks = [];
      for await (const chunk of stream) chunks.push(chunk);
      const buf = Buffer.concat(chunks);
      if (buf.length) {
        const ext = inner.fileName?.includes('.')
          ? '.' + inner.fileName.split('.').pop().toLowerCase()
          : ({ image: '.jpg', video: '.mp4', voice: '.ogg', audio: '.ogg', sticker: '.webp' }[kind] || '.bin');
        const name = `${Date.now()}-${String(ctx.stanzaId || 'q').replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 12)}${ext}`;
        const p = path.join(MEDIA_DIR, name);
        fs.writeFileSync(p, buf);
        out.quoted_media = [{ path: p, mime, kind }];
      } else {
        out.quoted_media = [{ path: '', mime, kind }]; // arrived, unseen
      }
    } catch (e) {
      log('quoted media download failed:', e.message);
      out.quoted_media = [{ path: '', mime, kind }]; // arrived, unseen
    }
  }
  return out;
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

// View-once media arrives wrapped (viewOnceMessage / V2 /
// V2Extension) or, in the older shape, as a media message carrying
// viewOnce=true.  Unwrap to the inner message so text/media
// extraction sees the payload, and report the flag — the personal
// surface (§15) preserves view-once media before it vanishes, and
// it can only preserve what the bridge admits it saw.
function unwrapViewOnce(m) {
  const msg = m.message || {};
  const inner = msg.viewOnceMessage?.message
    || msg.viewOnceMessageV2?.message
    || msg.viewOnceMessageV2Extension?.message;
  if (inner) return { msg: { ...m, message: inner }, viewOnce: true };
  const media = msg.imageMessage || msg.videoMessage;
  if (media && media.viewOnce) return { msg: m, viewOnce: true };
  return { msg: m, viewOnce: false };
}

const EXT_MIME = {
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
  '.webp': 'image/webp', '.gif': 'image/gif', '.mp4': 'video/mp4',
  '.mov': 'video/quicktime', '.mp3': 'audio/mpeg', '.ogg': 'audio/ogg',
  '.oga': 'audio/ogg', '.wav': 'audio/wav', '.m4a': 'audio/mp4',
  '.pdf': 'application/pdf',
};

async function downloadInbound(m) {
  // Returns {path, mime, kind} — path is "" when the media arrived
  // but its bytes could not be fetched (the payload must carry the
  // trace: media arrived, unseen; the Python side stamps pathless
  // entries media_unavailable) — or null when the message carries
  // no media at all.
  const kind = mediaKindOf(m);
  if (!kind) return null;
  const inner = m.message?.imageMessage || m.message?.videoMessage
    || m.message?.audioMessage || m.message?.documentMessage
    || m.message?.stickerMessage || {};
  const mime = inner.mimetype || 'application/octet-stream';
  try {
    const buf = await downloadMediaMessage(m, 'buffer', {});
    if (!buf || !buf.length) return { path: '', mime, kind };
    const ext = inner.fileName?.includes('.')
      ? '.' + inner.fileName.split('.').pop().toLowerCase()
      : ({ image: '.jpg', video: '.mp4', voice: '.ogg', audio: '.ogg', sticker: '.webp' }[kind] || '.bin');
    const name = `${Date.now()}-${(m.key.id || 'm').replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 12)}${ext}`;
    const p = path.join(MEDIA_DIR, name);
    fs.writeFileSync(p, buf);
    return { path: p, mime, kind };
  } catch (e) {
    log('media download failed:', e.message);
    return { path: '', mime, kind };
  }
}

// ── server-confirmed pairing codes ───────────────────────────────────
// Baileys' requestPairingCode() is a phantom-code machine (upstream
// issue #2737; verified 2026-11-04 against the newest published
// builds — 6.7.24 legacy and 7.0.0-rc14 both still do this, and
// master has no fix): lib/Socket/socket.js builds the code locally,
// fires the companion_hello IQ with sendNode — write-only, it
// resolves when the frame is WRITTEN, not when the server answers —
// and returns the code immediately.  A server refusal (the issue's
// captured 400 bad-request) lands afterwards, observed by nobody:
// the owner is handed a code WhatsApp never registered and the
// phone answers "Couldn't link device".
//
// So this bridge sends stage 1 itself: the node below is
// byte-identical to the one 6.7.24's requestPairingCode builds
// (same children, same order, same creds mutations — pairingCode
// + me set and persisted BEFORE the send, exactly as the library
// does), but it goes through sock.query(), which awaits the
// server's IQ result and throws on an error IQ
// (assertNodeErrorFree).  The returned promise therefore resolves
// ONLY with a code the server accepted at stage 1, and rejects
// with the server's own refusal text otherwise — the caller
// prints/broadcasts the code strictly after that resolution, so
// a refused request can never produce a displayable code.
// Stage 2 needs nothing from stage 1's response: when the phone
// enters the code, the server pushes the link_code_companion_reg
// notification that messages-recv.js completes from creds alone
// (pairingCode + pairingEphemeralKeyPair + signedIdentityKey) —
// all set here exactly as the library sets them.
// The platform triple is Baileys' own default for a socket
// created without a browser override (Browsers.ubuntu('Chrome'))
// — the same values 6.7.24's requestPairingCode reads out of its
// config for companion_platform_id / companion_platform_display.
const PAIRING_BROWSER = ['Ubuntu', 'Chrome', '22.04.4'];

async function requestPairingCodeConfirmed(sock, state, saveCreds,
                                           phoneNumber) {
  const { creds } = state;
  const pairingCode = bytesToCrockford(randomBytes(5));
  creds.pairingCode = pairingCode;
  creds.me = {
    id: jidEncode(phoneNumber, 's.whatsapp.net'),
    name: '~',
  };
  await saveCreds();
  // generatePairingKey, mirrored from 6.7.24: the code-derived
  // key wraps our pairing ephemeral public key, salt‖iv‖cipher.
  const salt = randomBytes(32);
  const randomIv = randomBytes(16);
  const key = await derivePairingCodeKey(pairingCode, salt);
  const ciphered = aesEncryptCTR(
    creds.pairingEphemeralKeyPair.public, key, randomIv);
  const wrappedEphemeralPub = Buffer.concat([salt, randomIv, ciphered]);
  await sock.query({
    tag: 'iq',
    attrs: {
      to: 's.whatsapp.net',
      type: 'set',
      xmlns: 'md',
    },
    content: [
      {
        tag: 'link_code_companion_reg',
        attrs: {
          jid: creds.me.id,
          stage: 'companion_hello',
          should_show_push_notification: 'true',
        },
        content: [
          {
            tag: 'link_code_pairing_wrapped_companion_ephemeral_pub',
            attrs: {},
            content: wrappedEphemeralPub,
          },
          {
            tag: 'companion_server_auth_key_pub',
            attrs: {},
            content: creds.noiseKey.public,
          },
          {
            tag: 'companion_platform_id',
            attrs: {},
            content: getPlatformId(PAIRING_BROWSER[1]),
          },
          {
            tag: 'companion_platform_display',
            attrs: {},
            content: `${PAIRING_BROWSER[1]} (${PAIRING_BROWSER[0]})`,
          },
          {
            tag: 'link_code_pairing_nonce',
            attrs: {},
            content: '0',
          },
        ],
      },
    ],
  });
  return pairingCode;
}

// ── WhatsApp connection ──────────────────────────────────────────────────────
let wa = null;
let meJid = '';
let meLid = ''; // the account's own LID (<lid>@lid) when it has one

async function connect() {
  const { state, saveCreds } = await useMultiFileAuthState(CREDS_DIR);
  const { version } = await fetchLatestBaileysVersion().catch(() => ({ version: undefined }));
  wa = makeWASocket({
    auth: state,
    printQRInTerminal: false,
    ...(version ? { version } : {}),
  });
  wa.ev.on('creds.update', saveCreds);

  // Pairing-code stability: Baileys rotates the qr ref roughly every
  // 20s while unregistered, and EVERY pairing-code request mints a
  // new code and kills the previous one.  Requesting on every qr
  // event therefore invalidated the code the owner was typing
  // mid-entry (WhatsApp: "Couldn't link device").  One request per
  // socket session: these flags live in this connect() closure, so a
  // reconnect (new socket) re-arms exactly one request, and 'open'
  // or 'close' clears them — never a second request while a code
  // is outstanding on this socket.  And the one request is the
  // server-confirmed one above, never Baileys' phantom-returning
  // requestPairingCode: nothing is printed until WhatsApp has
  // accepted the code at stage 1.
  let pairingCodeRequested = false;
  let pairingCodeIssued = false;

  wa.ev.on('connection.update', (u) => {
    const { connection, lastDisconnect, qr } = u;
    if (qr) {
      if (PAIRING_CODE && PAIRING_NUMBER) {
        // Pairing-code login: an 8-char code the owner types into the phone.
        if (!pairingCodeRequested) {
          pairingCodeRequested = true;
          requestPairingCodeConfirmed(wa, state, saveCreds,
                                      PAIRING_NUMBER).then(
            (code) => {
              pairingCodeIssued = true;
              log(`PAIRING CODE for +${PAIRING_NUMBER}: ${code} (type into WhatsApp > Linked Devices)`);
              broadcast({ type: 'pairing_code', data: code });
            },
            (e) => log('pairing code request failed:', e?.message || e),
          );
        }
        // else: a code is already outstanding on this socket — the
        // rotating qr ref is NOT a reason to mint a replacement.
      } else {
        log('QR received — scan with WhatsApp (Linked Devices).');
        try { qrcode.generate(qr, { small: true }); } catch { /* ignore */ }
        broadcast({ type: 'qr', data: qr });
      }
    }
    if (connection === 'open') {
      pairingCodeRequested = false;
      pairingCodeIssued = false;
      meJid = jidNormalizedUser(wa.user?.id || '');
      // Baileys exposes the account's own LID on the same user
      // object (creds.me.lid) for LID-migrated accounts — the
      // adapter needs it to recognise the self-chat, which
      // WhatsApp addresses by LID there, not by phone JID.
      meLid = wa.user?.lid ? jidNormalizedUser(wa.user.lid) : '';
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
      // Presence re-assert (§15): a wanted presence outlives
      // the socket that carried it — re-send it on every open
      // so always-online survives reconnects with no Python
      // beat having to notice first. Best-effort: the beat
      // remains the backstop.
      if (lastPresenceState) {
        wa.sendPresenceUpdate(lastPresenceState).catch(
          (e) => log('presence re-assert on open failed:',
                     e?.message || e));
      }
    }
    if (connection === 'close') {
      const code = lastDisconnect?.error?.output?.statusCode;
      // A pairing WhatsApp itself ended: a code was issued on this
      // socket, the link never landed (creds unregistered — a
      // registered socket's routine 515 restart / later logout is
      // not this), and the close is terminal under the policy
      // table (STOP: 401/500/440/411/403 — the bridge parks, no
      // fresh code is coming).  Name it on its own line so the
      // supervisor can tell the owner the truth and stop showing
      // the dead code.  Transient closes (RETRY: 428/408/503 and
      // 515 restartRequired) stay silent here — the reconnect
      // issues a fresh code on the new socket, which supersedes.
      if (PAIRING_CODE && pairingCodeIssued && !state.creds.registered) {
        const verdict = classifyDisconnect(code);
        if (verdict.action === ACTION.STOP) {
          log(`PAIRING LINK FAILED: ${verdict.label} (code ${code ?? 'n/a'})`);
        }
      }
      pairingCodeRequested = false;
      pairingCodeIssued = false;
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

  // Incoming call offers (chat & privacy wave): forwarded,
  // never acted on here — rejection is the owner's anticall
  // toggle, decided on the Python side like every other
  // personal-surface act.  Baileys is a signaling observer: it
  // can see offers and reject them, nothing more.
  wa.ev.on('call', (calls) => {
    try {
      for (const c of calls || []) {
        if (!c || c.status !== 'offer') continue;
        broadcast({ type: 'call_offer', id: c.id || '',
          from: jidNormalizedUser(c.from || c.chatId || ''),
          ts: c.date ? new Date(c.date).getTime() : Date.now() });
      }
    } catch (e) {
      log('call handler error:', e?.message || e);
    }
  });

  wa.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify') return;
    for (const m of messages) {
      try {
        if (!m.message) continue;
        const jid = jidNormalizedUser(m.key.remoteJid || '');
        if (!jid) continue;
        const actorId = m.key.participant
          ? jidNormalizedUser(m.key.participant) : jid;
        const tsMs = Number(m.messageTimestamp || Date.now() / 1000) * 1000;
        // Protocol messages are not conversation: a REVOKE (type 0)
        // is a delete-for-everyone, a MESSAGE_EDIT (type 14) is an
        // edit.  Broadcast them as the evidentiary feeds the Python
        // adapter already carries ({type:'revoke'|'edit'}) — group
        // defense sensors and the DM antidelete surface both read
        // from these, and neither can see what the bridge drops.
        const proto = m.message?.protocolMessage;
        if (proto) {
          const targetId = proto.key?.id || '';
          if (targetId && proto.type === 0) {
            broadcast({ type: 'revoke', id: targetId,
              chat: { id: jid, kind: kindOf(jid) },
              from: { id: actorId }, ts: tsMs });
          } else if (targetId && proto.type === 14) {
            broadcast({ type: 'edit', id: targetId,
              chat: { id: jid, kind: kindOf(jid) },
              from: { id: actorId, name: m.pushName || '' },
              text: textOf({ ...m, message: proto.editedMessage || {} }),
              ts: tsMs });
          }
          continue;
        }
        const unwrapped = unwrapViewOnce(m);
        const mm = unwrapped.msg;
        // Status posts: not conversation either — they are the
        // status-saving surface's feed (§15).  The account's OWN
        // statuses are skipped (saving a man's own posts back to
        // him is noise); everyone else's ride out as status_post
        // events with their media already downloaded.
        if (jid === 'status@broadcast') {
          if (m.key.fromMe) continue;
          const media = await downloadInbound(mm);
          broadcast({ type: 'status_post', id: m.key.id || '',
            from: { id: actorId, name: m.pushName || '' },
            text: textOf(mm), media: media ? [media] : [],
            view_once: unwrapped.viewOnce, ts: tsMs });
          continue;
        }
        // Own outgoing messages (shared-account mode): forward as
        // informational own_message events so the bot can track the
        // owner's replies (double-reply discipline, per-chat profile).
        // Never processed as inbound — the brain stays silent on them.
        if (m.key.fromMe) {
          const ownOut = {
            type: 'own_message',
            id: m.key.id || '',
            chat: { id: jid, kind: kindOf(jid), title: chatTitles.get(jid) || '' },
            text: textOf(mm),
            reply_to: replyToOf(mm),
            view_once: unwrapped.viewOnce,
            ts: tsMs,
            ...(await quotedPayload(mm)),
          };
          remember(jid, { ...ownOut, incoming: false, sender: '', key: m.key });
          broadcast(ownOut);
          continue;
        }
        const senderId = actorId;
        const kind = kindOf(jid);
        let title = chatTitles.get(jid) || '';
        if (kind === 'group' && !title) {
          try {
            const meta = await wa.groupMetadata(jid);
            title = meta.subject || '';
            chatTitles.set(jid, title);
          } catch { /* ignore */ }
        }
        if (kind === 'channel' && !title) {
          // A channel post's author IS the channel — no push
          // name to fall back to. The newsletter metadata
          // carries the channel's real name; cache it like a
          // group's subject. Guarded: a Baileys build without
          // the verb leaves the title to the JID fallback.
          try {
            if (typeof wa.newsletterMetadata === 'function') {
              const meta = await wa.newsletterMetadata('jid', jid);
              title = (meta && meta.name) || '';
              if (title) chatTitles.set(jid, title);
            }
          } catch { /* ignore */ }
        }
        if (!title) title = m.pushName || senderId.split('@')[0];
        else if (kind === 'dm') chatTitles.set(jid, m.pushName || title);
        const media = await downloadInbound(mm);
        // @-mention of this account (groups): drives the brain's
        // should-speak-in-group exactly like Telegram mentions.
        const mentioned = mentionedJidOf(mm).some(
          (j) => jidNormalizedUser(String(j)) === meJid);
        const out = {
          type: 'message',
          id: m.key.id || '',
          chat: { id: jid, kind, title },
          from: { id: senderId, name: m.pushName || '' },
          text: textOf(mm),
          media: media ? [media] : [],
          reply_to: replyToOf(mm),
          mentioned,
          // The full mention payload (chat & privacy wave — the
          // antitag sensor counts it; additive, the Python
          // envelope already reads a "mentions" list).
          mentions: mentionedJidOf(mm).map(String),
          view_once: unwrapped.viewOnce,
          ts: tsMs,
          ...(await quotedPayload(mm)),
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

async function cmdSend({ chat, text, reply_to, buttons, mentions }) {
  const opts = {};
  if (reply_to && msgCache.has(reply_to)) {
    opts.quoted = msgCache.get(reply_to); // raw WAMessage -> quoted reply
  }
  // Inline buttons ride the same contract as Telegram: rows of
  // [label, command]. WhatsApp's legacy button messages cap at 3
  // buttons — the first three render natively, the rest degrade
  // to text action lines in the same message; if WhatsApp refuses
  // the button message entirely (increasingly common), ALL of
  // them degrade to text. The actions are never silently dropped.
  const flat = [];
  for (const row of buttons || []) {
    for (const b of row || []) {
      if (Array.isArray(b) && b.length >= 2) {
        flat.push([String(b[0]), String(b[1])]);
      }
    }
  }
  if (flat.length) {
    const asLines = (items) => items
      .map(([label, id]) => `• ${label} — /${id}`).join('\n');
    const native = flat.slice(0, 3).map(([label, id]) => ({
      buttonId: id,
      buttonText: { displayText: label.slice(0, 25) },
      type: 1,
    }));
    let body = text || '';
    if (flat.length > 3) {
      body += (body ? '\n' : '') + asLines(flat.slice(3));
    }
    try {
      return await wa.sendMessage(
        chat, { text: body, buttons: native, headerType: 1 }, opts);
    } catch {
      const lines = asLines(flat);
      return wa.sendMessage(
        chat, { text: (text || '') + (text ? '\n' : '') + lines },
        opts);
    }
  }
  // Mentions (chat & privacy wave — /tagall): Baileys takes the
  // JID list natively; the names need not appear in the text.
  const payload = { text: text || '' };
  if (Array.isArray(mentions) && mentions.length) {
    payload.mentions = mentions.map(String);
  }
  return wa.sendMessage(chat, payload, opts);
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

// The anchor Baileys' archive / mark-read patches need: the
// chat's last message as a WAMessage (its raw cached form when
// the cache holds it, a key-only shell otherwise).  No message
// in bridge memory → null, and the caller refuses honestly —
// the same memory discipline as message_delete.
function lastMessagesFor(jid) {
  const arr = recentByChat.get(jid) || [];
  for (let i = arr.length - 1; i >= 0; i--) {
    const entry = arr[i];
    if (entry && entry.key && entry.key.id) {
      return [msgCache.get(entry.key.id) || { key: entry.key }];
    }
  }
  return null;
}

async function handleCommand(sock, req) {
  const { id, cmd } = req;
  if (!wa) return reply(sock, id, { ok: false, error: 'whatsapp not connected yet' });
  try {
    switch (cmd) {
      case 'send': {
        const sent = await cmdSend(req);
        // The real WhatsApp message id — the Python adapter keys
        // delivery receipts off it. (The reply's own "id" field is
        // just the request id echoed back.)
        return reply(sock, id, { ok: true, message_id: sent?.key?.id || '' });
      }
      case 'send_media': {
        const sent = await cmdSendMedia(req);
        return reply(sock, id, { ok: true, message_id: sent?.key?.id || '' });
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
          id: (m.key && m.key.id) || m.id || '',
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
      case 'presence': {
        // Account presence (§15 always-online): Baileys
        // sendPresenceUpdate with NO chat jid sets the global
        // presence ('available' holds the account online;
        // 'unavailable' releases it).  Errors surface honestly —
        // unlike 'read', a presence that didn't land must not ack.
        const state = req.state === 'unavailable' ? 'unavailable' : 'available';
        await wa.sendPresenceUpdate(state);
        // Remember the want: the connection.update 'open'
        // handler re-asserts it after every reconnect.
        lastPresenceState = state;
        return reply(sock, id, { ok: true });
      }
      case 'status_send': {
        // Post a status AS the account owner (§15 status
        // posting — the send half of the status family).
        // Baileys shape: a send to status@broadcast whose
        // fan-out is the statusJidList — without one Baileys
        // encrypts for nobody, so the audience is explicit:
        // the request's list when given, else every DM
        // counterpart this bridge has seen this session. No
        // audience at all is an honest refusal, never a post
        // nobody can see. (The Go engine needs no list:
        // whatsmeow resolves the account's server-side status
        // recipients itself.)
        const text = String(req.text || '').trim();
        const mediaPath = String(req.path || '').trim();
        if (!text && !mediaPath) {
          return reply(sock, id, { ok: false, error: 'status_send needs text or a media path' });
        }
        let audience = Array.isArray(req.audience)
          ? req.audience.map(String).filter(Boolean) : [];
        if (!audience.length) {
          audience = [...chatTitles.keys()].filter(
            (j) => j.endsWith('@s.whatsapp.net') && j !== meJid);
        }
        if (!audience.length) {
          return reply(sock, id, { ok: false, error: 'no status audience known — this bridge has not seen any contacts this session, and none was given' });
        }
        const opts = { statusJidList: audience };
        let content;
        if (mediaPath) {
          let buf;
          try {
            buf = fs.readFileSync(mediaPath);
          } catch {
            return reply(sock, id, { ok: false, error: 'media file not found: ' + mediaPath });
          }
          const caption = String(req.caption || '') || text;
          if (['.mp4', '.mov', '.gif'].includes(extOf(mediaPath))) {
            content = { video: buf, caption };
          } else if (['.jpg', '.jpeg', '.png', '.webp'].includes(extOf(mediaPath))) {
            content = { image: buf, caption };
          } else {
            return reply(sock, id, { ok: false, error: 'status media must be an image or a video: ' + mediaPath });
          }
        } else {
          content = { text };
          opts.backgroundColor = '#000000';
          opts.font = 2;
        }
        const sent = await wa.sendMessage('status@broadcast', content, opts);
        return reply(sock, id, { ok: true, message_id: sent?.key?.id || '' });
      }
      case 'status_view': {
        // Mark one contact status viewed (§15 status viewing):
        // the read receipt for a status post is a readMessages
        // call with the status key — status@broadcast remoteJid,
        // the poster as participant.
        if (!req.id || !req.from) {
          return reply(sock, id, { ok: false, error: 'status_view needs the status message id and the poster jid' });
        }
        await wa.readMessages([{ remoteJid: 'status@broadcast', id: String(req.id), participant: String(req.from), fromMe: false }]);
        return reply(sock, id, { ok: true });
      }
      case 'edit': {
        // Edit one of the account's OWN messages (the Python
        // adapter's edit_message): Baileys edits by re-sending
        // with the original key.  The key comes from the bridge's
        // own message cache — a message it never saw, or one
        // that isn't the account's own, is an honest refusal.
        // WhatsApp's ~15-minute edit window is enforced
        // server-side and its refusal surfaces as the error.
        const jid = jidNormalizedUser(req.chat || '');
        const mid = String(req.message_id || '').trim();
        if (!mid) return reply(sock, id, { ok: false, error: 'edit needs a message_id' });
        const cached = msgCache.get(mid);
        if (!cached || !cached.key) {
          return reply(sock, id, { ok: false, error: 'edit target is not in bridge memory — only messages this bridge has seen can be edited' });
        }
        if (!cached.key.fromMe) {
          return reply(sock, id, { ok: false, error: 'only the account\'s own messages can be edited' });
        }
        await wa.sendMessage(jid, { text: String(req.text || ''), edit: cached.key });
        return reply(sock, id, { ok: true, message_id: mid });
      }
      case 'chat_modify': {
        // Chat-state ops (chat & privacy wave): Baileys
        // chatModify — archive / pin / mute / mark read-unread.
        // Tri-state like the Go bridge: only present keys apply.
        const jid = jidNormalizedUser(req.chat || '');
        let applied = 0;
        if ('archive' in req || 'read' in req) {
          const last = lastMessagesFor(jid);
          if (!last) {
            return reply(sock, id, { ok: false, error: 'no message from that chat in bridge memory to anchor the change' });
          }
          if ('archive' in req) {
            await wa.chatModify({ archive: !!req.archive, lastMessages: last }, jid);
            applied++;
          }
          if ('read' in req) {
            await wa.chatModify({ markRead: !!req.read, lastMessages: last }, jid);
            applied++;
          }
        }
        if ('pin' in req) {
          await wa.chatModify({ pin: !!req.pin }, jid);
          applied++;
        }
        if ('mute_seconds' in req) {
          const secs = Number(req.mute_seconds || 0);
          // Baileys passes the value through as the mute END
          // timestamp in seconds; null unmutes.
          const mute = secs === 0 ? null
            : Math.floor(Date.now() / 1000)
              + (secs < 0 ? 10 * 365 * 86400 : secs);
          await wa.chatModify({ mute }, jid);
          applied++;
        }
        if (!applied) {
          return reply(sock, id, { ok: false, error: 'chat_modify needs at least one of: archive, pin, mute_seconds, read' });
        }
        return reply(sock, id, { ok: true });
      }
      case 'message_star': {
        const jid = jidNormalizedUser(req.chat || '');
        const mid = String(req.message_id || '').trim();
        if (!mid) return reply(sock, id, { ok: false, error: 'message_star needs a message_id' });
        const cached = msgCache.get(mid);
        if (!cached || !cached.key) {
          return reply(sock, id, { ok: false, error: 'message not in bridge memory — its star key cannot be built' });
        }
        await wa.chatModify({
          star: {
            messages: [{ id: mid, fromMe: !!cached.key.fromMe }],
            star: !!req.star,
          },
        }, jid);
        return reply(sock, id, { ok: true });
      }
      case 'status_react': {
        // React to one contact status post as the owner (the
        // status_react toggle's verb): a reaction keyed on the
        // status message at status@broadcast.
        if (!req.id || !req.from || !req.emoji) {
          return reply(sock, id, { ok: false, error: 'status_react needs the status message id, the poster jid and an emoji' });
        }
        await wa.sendMessage('status@broadcast', {
          react: {
            text: String(req.emoji),
            key: { remoteJid: 'status@broadcast', id: String(req.id), participant: String(req.from), fromMe: false },
          },
        });
        return reply(sock, id, { ok: true });
      }
      case 'reject_call': {
        if (!req.id || !req.from) {
          return reply(sock, id, { ok: false, error: 'reject_call needs the call id and the caller jid' });
        }
        await wa.rejectCall(String(req.id), String(req.from));
        return reply(sock, id, { ok: true });
      }
      case 'blocklist': {
        const jid = jidNormalizedUser(req.jid || '');
        if (!jid) return reply(sock, id, { ok: false, error: 'blocklist needs a jid' });
        if (req.action !== 'block' && req.action !== 'unblock') {
          return reply(sock, id, { ok: false, error: 'blocklist action must be block or unblock' });
        }
        await wa.updateBlockStatus(jid, req.action);
        return reply(sock, id, { ok: true });
      }
      case 'chat_disappear': {
        // Disappearing-messages timer: Baileys carries the verb
        // for GROUPS only (groupToggleEphemeral).  DM timers
        // ride the Go bridge engine — this rung refuses
        // honestly instead of pretending.
        const jid = jidNormalizedUser(req.chat || '');
        const secs = Number(req.seconds);
        if (!Number.isFinite(secs) || secs < 0) {
          return reply(sock, id, { ok: false, error: 'chat_disappear needs seconds >= 0 (0 turns disappearing messages off)' });
        }
        if (!jid.endsWith('@g.us')) {
          return reply(sock, id, { ok: false, error: 'this Baileys build can only set disappearing timers in groups — DM timers ride the Go bridge engine' });
        }
        await wa.groupToggleEphemeral(jid, secs);
        return reply(sock, id, { ok: true });
      }
      case 'check_number': {
        // The one registration question WhatsApp publishes
        // (banchecker's number signal).  Nothing about bans is
        // returned because WhatsApp publishes nothing about bans.
        const digits = String(req.phone || '').replace(/[^\d]/g, '');
        if (!digits) return reply(sock, id, { ok: false, error: 'check_number needs a phone number' });
        const rows = await wa.onWhatsApp(`${digits}@s.whatsapp.net`);
        const row = Array.isArray(rows) ? rows[0] : null;
        if (!row) return reply(sock, id, { ok: false, error: 'WhatsApp returned no answer for that number' });
        return reply(sock, id, { ok: true, exists: !!row.exists, jid: row.jid || '' });
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
        return reply(sock, id, { ok: true, user: meJid, lid: meLid, state: wa ? 'open' : 'closed' });
      }
      default:
        // workstream 3: group/community commands (read-only) before "unknown cmd",
        // then the mutating admin commands (owner-confirmed on the Python side).
        return (await registerGroupCommands({ getSocket: () => wa, reply, log })(sock, req))
          || (await registerGroupAdminCommands({ getSocket: () => wa, reply, log, getCachedMessage: (id) => msgCache.get(id) })(sock, req))
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
