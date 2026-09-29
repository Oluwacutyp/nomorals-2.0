/**
 * NoMorals WhatsApp bridge.
 *
 * Holds the WhatsApp Web session (Baileys) and exposes it to the Python core
 * over a localhost JSON-lines TCP socket. One JSON object per line, both ways.
 *
 *   bridge -> python : {type: status|qr|message|response, ...}
 *   python -> bridge : {id, cmd: send|typing|history|status, ...}
 *
 * Credential state lives in .creds/ (multi-file Baileys auth state). That
 * directory is the session — treat it like a password: never commit it,
 * never back it up to a shared git repo.
 *
 * Run:   npm install && npm start
 * Env:   WA_BRIDGE_PORT (8787), WA_MEDIA_DIR (data/whatsapp_media)
 *
 * Note on ToS: WhatsApp does not offer an official personal-account API.
 * This bridge drives a linked device on your own account. Keep volume
 * human; the Python core enforces rate limits and autonomy caps.
 *
 * Self-chat ("Message Yourself"): messages the account sends to itself are
 * forwarded to the core as inbound, so the owner can talk to the companion
 * from the same phone. The bridge's own replies are filtered by message id
 * (loop guard).
 */

import net from "node:net";
import fs from "node:fs";
import path from "node:path";
import {
  makeWASocket,
  useMultiFileAuthState,
  makeCacheableSignalKeyStore,
  downloadMediaMessage,
  fetchLatestBaileysVersion,
  DisconnectReason,
} from "@whiskeysockets/baileys";
import pino from "pino";

let qrTerminal = null;
try {
  qrTerminal = (await import("qrcode-terminal")).default;
} catch {
  // optional: QR still travels over the protocol
}
let qrPng = null;
try {
  qrPng = (await import("qrcode")).default;
} catch {
  // optional: full-size PNG written next to the bridge (npm i qrcode)
}
const QR_FILE = path.resolve("latest-qr.png");

const PORT = parseInt(process.env.WA_BRIDGE_PORT || "8787", 10);
const HOST = process.env.WA_BRIDGE_HOST || "127.0.0.1";
const MEDIA_DIR = process.env.WA_MEDIA_DIR || path.resolve("data/whatsapp_media");
const CREDS_DIR = process.env.WA_CREDS_DIR || path.resolve(".creds");
fs.mkdirSync(MEDIA_DIR, { recursive: true });

const log = pino({ level: "silent" });
const clients = new Set();

function broadcast(obj) {
  const line = JSON.stringify(obj) + "\n";
  for (const c of clients) {
    try {
      c.write(line);
    } catch {
      clients.delete(c);
    }
  }
}

/** jid -> friendly kind */
function kindOf(jid) {
  if (typeof jid !== "string") return "dm";
  if (jid.endsWith("@g.us")) return "group";
  if (jid.endsWith("@broadcast")) return "channel";
  return "dm";
}

function normalizeJid(jid) {
  if (typeof jid !== "string") return jid;
  if (jid.includes("@")) return jid;
  // bare number -> DM jid
  return jid.replace(/[^0-9]/g, "") + "@c.us";
}

let socket = null;
// own JID once connected (e.g. "2348...@s.whatsapp.net") — identifies the
// "Message Yourself" self-chat
let ownJid = null;
// ids of messages this bridge sent — the self-chat loop guard
const sentIds = new Set();

function connectionState() {
  if (!socket || !socket.ws) return "closed";
  return socket.ws.readyState === 1 ? "open" : "closed";
}

async function handle(cmd, writer) {
  const id = cmd.id || "";
  const reply = (ok, extra = {}) => {
    try {
      writer.write(JSON.stringify({ id, ok, ...extra }) + "\n");
    } catch {
      /* client went away */
    }
  };
  if (!socket) return reply(false, { error: "not connected" });

  switch (cmd.cmd) {
    case "status":
      return reply(true, {
        state: connectionState(),
        user: socket.user?.id || null,
      });

    case "send": {
      try {
        const jid = normalizeJid(cmd.chat);
        const payload = { text: String(cmd.text || "") };
        if (cmd.reply_to) payload.quoted = undefined; // quoting needs the quoted message object; kept as a no-op
        const res = await socket.sendMessage(jid, payload);
        const mid = res?.key?.id;
        if (mid) {
          sentIds.add(mid);
          if (sentIds.size > 500) sentIds.clear(); // guard only needs recent ids
        }
        return reply(true, { id: mid || "sent" });
      } catch (e) {
        return reply(false, { error: String(e?.message || e) });
      }
    }

    case "typing": {
      try {
        const jid = normalizeJid(cmd.chat);
        const composing = Number(cmd.seconds) > 0;
        if (!composing) {
          await socket.sendPresenceUpdate("paused", jid);
          return reply(true);
        }
        // WhatsApp's composing presence decays after ~10s, so the indicator
        // must be re-sent to survive the requested duration. Cap at 58s
        // (WhatsApp's own presence ceiling).
        const seconds = Math.min(58, Math.max(1, Number(cmd.seconds) || 3));
        await socket.sendPresenceUpdate("composing", jid);
        const stopAt = Date.now() + seconds * 1000;
        const refresh = setInterval(() => {
          if (Date.now() >= stopAt || !socket) {
            clearInterval(refresh);
            socket?.sendPresenceUpdate("paused", jid).catch(() => {});
            return;
          }
          socket.sendPresenceUpdate("composing", jid).catch(() => {});
        }, 8000);
        refresh.unref?.();
        return reply(true);
      } catch (e) {
        return reply(false, { error: String(e?.message || e) });
      }
    }

    case "history": {
      try {
        const jid = normalizeJid(cmd.chat);
        const limit = Math.min(100, Math.max(1, Number(cmd.limit) || 20));
        if (typeof socket.fetchMessages !== "function") {
          return reply(true, { messages: [] });
        }
        const msgs = await socket.fetchMessages(jid);
        const out = (msgs || [])
          .filter((m) => m && m.message)
          .slice(-limit)
          .map((m) => ({
            text: m.message?.conversation || m.message?.extendedTextMessage?.text || "",
            sender: m.key?.fromMe ? "me" : m.participant || m.key?.remoteJid,
            ts: (m.messageTimestamp || 0) * 1000,
          }));
        return reply(true, { messages: out });
      } catch (e) {
        return reply(false, { error: String(e?.message || e) });
      }
    }

    default:
      return reply(false, { error: `unknown cmd: ${cmd.cmd}` });
  }
}

async function onUpsert({ messages }) {
  for (const m of messages || []) {
    try {
      if (!m || !m.key) continue;
      const jid = m.key.remoteJid;
      if (!jid || jid === "status@broadcast") continue;
      // Self-chat ("Message Yourself"): messages the user sends to themself
      // arrive with fromMe=true. Accept only the self-chat, and skip the
      // bridge's own replies (loop guard).
      if (m.key.fromMe) {
        if (jid !== ownJid) continue;
        if (sentIds.has(m.key.id)) continue;
      }

      let text =
        m.message?.conversation ||
        m.message?.extendedTextMessage?.text ||
        m.message?.imageMessage?.caption ||
        m.message?.videoMessage?.caption ||
        m.message?.documentMessage?.caption ||
        "";

      const media = [];
      const kind = kindOf(jid);
      for (const [key, name] of [
        ["imageMessage", "image"],
        ["videoMessage", "video"],
        ["audioMessage", "audio"],
        ["documentMessage", "document"],
      ]) {
        const msg = m.message?.[key];
        if (!msg) continue;
        const ext =
          (msg.mimetype || "").split("/")[1] ||
          (key === "audioMessage" ? "ogg" : key === "documentMessage" ? "bin" : "jpg");
        const file = path.join(
          MEDIA_DIR,
          `${Date.now()}-${Math.random().toString(36).slice(2, 8)}.${ext}`
        );
        try {
          await downloadMediaMessage(m, "base64", { filename: file });
          media.push({ path: file, mime: msg.mimetype || "", kind: name });
        } catch (e) {
          log.info({ err: String(e) }, "media download failed");
        }
      }

      if (!text && !media.length) continue;
      broadcast({
        type: "message",
        chat: { id: jid, kind, title: kind === "group" ? "group" : "" },
        from: { id: m.participant || jid, name: m.pushName || "" },
        text,
        media,
        reply_to: m.message?.extendedTextMessage?.contextInfo?.stanzaId || "",
        ts: (m.messageTimestamp || 0) * 1000,
      });
    } catch (e) {
      log.info({ err: String(e) }, "message upsert failed");
    }
  }
}

async function main() {
  const { state, saveCreds } = await useMultiFileAuthState(CREDS_DIR);
  let version;
  try {
    ({ version } = await fetchLatestBaileysVersion());
  } catch {
    version = undefined; // fall back to Baileys default
  }

  socket = makeWASocket({
    version,
    logger: log,
    printQRInTerminal: false,
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, log),
    },
    syncFullHistory: false,
    markOnlineOnConnect: true,
    generateHighQualityLinkPreview: false,
  });

  socket.ev.on("creds.update", saveCreds);
  socket.ev.on("connection.update", (update) => {
    if (update.qr) {
      broadcast({ type: "qr", data: update.qr });
      // Full-size terminal QR (small ones decode too slowly — the code
      // rotates ~every 20s and phone cameras usually expire mid-scan),
      // plus a PNG written to disk for easy full-screen scanning.
      if (qrTerminal) {
        console.log("Scan this QR with WhatsApp -> Linked Devices:");
        qrTerminal.generate(update.qr);
      }
      if (qrPng) {
        qrPng.toFile(QR_FILE, update.qr, { width: 640, margin: 2 })
          .then(() => console.log(`QR image: ${QR_FILE}  (open it and scan from another screen)`))
          .catch(() => {});
      }
    }
    if (update.connection) {
      broadcast({
        type: "status",
        state: update.connection,
        user: socket?.user?.id || null,
      });
      if (update.connection === "open") {
        ownJid = socket?.user?.id || null;
        console.log(`whatsapp bridge: linked as ${socket?.user?.id || "?"}`);
        console.log("tip: you can talk to the companion by messaging yourself in WhatsApp");
      } else if (update.reason && update.reason !== DisconnectReason.loggedOut) {
        console.log(`whatsapp bridge: ${update.connection} (${update.reason}); reconnecting...`);
      }
    }
  });
  socket.ev.on("messages.upsert", onUpsert);

  const server = net.createServer((client) => {
    clients.add(client);
    let buf = "";
    client.on("data", (chunk) => {
      buf += chunk.toString("utf-8");
      let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!line) continue;
        let cmd;
        try {
          cmd = JSON.parse(line);
        } catch {
          client.write(JSON.stringify({ ok: false, error: "bad json" }) + "\n");
          continue;
        }
        handle(cmd, client).catch((e) => {
          client.write(JSON.stringify({ id: cmd.id || "", ok: false, error: String(e?.message || e) }) + "\n");
        });
      }
    });
    client.on("close", () => clients.delete(client));
    client.on("error", () => clients.delete(client));
  });

  server.listen(PORT, HOST, () => {
    console.log(`whatsapp bridge listening on ${HOST}:${PORT}`);
    console.log(`media dir: ${MEDIA_DIR}`);
  });

  process.on("SIGINT", () => {
    console.log("\nshutting down...");
    socket?.ev?.emit?.("disconnect");
    process.exit(0);
  });
}

main().catch((e) => {
  console.error("bridge failed to start:", e);
  process.exit(1);
});
