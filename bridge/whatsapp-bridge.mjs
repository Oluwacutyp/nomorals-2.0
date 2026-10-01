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
 *     {"type":"message","chat":{"id","kind","title"},"from":{"id","name"},
 *      "text":"...","media":[{"path","mime","kind"}],"reply_to":"","ts":<ms>}
 *     {"id":"<req>","ok":true|false,"error":"", ...extra}
 *
 *   python -> bridge
 *     {"id":"<req>","cmd":"send","chat":"<jid>","text":"...","reply_to":""}
 *     {"id":"<req>","cmd":"send_media","chat":"<jid>","path":"...","caption":"...","ptt":false}
 *     {"id":"<req>","cmd":"typing","chat":"<jid>","seconds":3}
 *     {"id":"<req>","cmd":"history","chat":"<jid>","limit":20}
 *     {"id":"<req>","cmd":"read","chat":"<jid>"}
 *     {"id":"<req>","cmd":"chats","limit":30}
 *     {"id":"<req>","cmd":"status"}
 *
 * Run:  node whatsapp-bridge.mjs [--port 8787] [--creds ./.creds] [--media ./media]
 * First run prints a QR code in the terminal — scan it with WhatsApp
 * (Settings -> Linked Devices -> Link a Device). The session persists in
 * .creds/ afterwards; guard that directory like a password.
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

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};
const PORT = parseInt(opt('--port', process.env.WA_BRIDGE_PORT || '8787'), 10);
const CREDS_DIR = path.resolve(here, opt('--creds', process.env.WA_CREDS_DIR || './.creds'));
const MEDIA_DIR = path.resolve(here, opt('--media', process.env.WA_MEDIA_DIR || './media'));
fs.mkdirSync(MEDIA_DIR, { recursive: true });

const log = (...a) => console.log(new Date().toISOString(), '[bridge]', ...a);

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
      log('QR received — scan with WhatsApp (Linked Devices).');
      try { qrcode.generate(qr, { small: true }); } catch { /* ignore */ }
      broadcast({ type: 'qr', data: qr });
    }
    if (connection === 'open') {
      meJid = jidNormalizedUser(wa.user?.id || '');
      log('connected as', meJid);
      broadcast({ type: 'status', state: 'open', user: meJid });
    }
    if (connection === 'close') {
      const code = lastDisconnect?.error?.output?.statusCode;
      const loggedOut = code === DisconnectReason.loggedOut;
      log('connection closed', loggedOut ? '(logged out — delete .creds/ and re-scan)' : `(code ${code}, retrying)`);
      broadcast({ type: 'status', state: 'closed', user: meJid });
      if (!loggedOut) setTimeout(connect, 5000);
    }
  });

  wa.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify') return;
    for (const m of messages) {
      try {
        if (!m.message || m.key.fromMe) continue;
        const jid = jidNormalizedUser(m.key.remoteJid || '');
        if (!jid || jid === 'status@broadcast') continue;
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
        const out = {
          type: 'message',
          chat: { id: jid, kind, title },
          from: { id: senderId, name: m.pushName || '' },
          text: textOf(m),
          media: media ? [media] : [],
          reply_to: replyToOf(m),
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
        await wa.sendPresenceUpdate('composing', req.chat);
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
        return reply(sock, id, { ok: false, error: `unknown cmd: ${cmd}` });
    }
  } catch (e) {
    return reply(sock, id, { ok: false, error: e.message || String(e) });
  }
}

// ── TCP server ───────────────────────────────────────────────────────────────
const server = net.createServer((sock) => {
  clients.add(sock);
  // send current state immediately so a fresh client knows where we stand
  sock.write(JSON.stringify({ type: 'status', state: meJid ? 'open' : 'closed', user: meJid }) + '\n');
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
  connect().catch((e) => { log('connect failed:', e.message); process.exit(1); });
});

process.on('SIGINT', () => process.exit(0));
process.on('SIGTERM', () => process.exit(0));
