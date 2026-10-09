#!/usr/bin/env node
/* =============================================================================
 * call-spike.mjs — WhatsApp audio-call feasibility spike for Baileys 6.7.24
 * (WORKSTREAM 2: RESEARCH + SPIKE ONLY. NOT production integration.)
 *
 * ── Findings (verified 2026-10-09 against the installed package) ─────────────
 *
 * 1) Baileys 6.7.24 = call-SIGNALING OBSERVER only. Sources:
 *    - bridge/node_modules/@whiskeysockets/baileys/lib/Socket/index.d.ts:49
 *      → the ONLY call action exposed is
 *        rejectCall: (callId: string, callFrom: string) => Promise<void>;
 *      There is NO acceptCall, NO placeCall, NO hangupCall.
 *    - bridge/node_modules/@whiskeysockets/baileys/lib/Types/Call.d.ts
 *      → WACallEvent { chatId, from, id, date, status, isVideo?, isGroup?,
 *        offline, latencyMs? }; WACallUpdateType =
 *        'offer' | 'ringing' | 'timeout' | 'reject' | 'accept' | 'terminate'.
 *    - bridge/node_modules/@whiskeysockets/baileys/lib/Socket/messages-recv.js
 *      handleCall() (l.~755) parses raw <call> stanzas and does
 *      ev.emit('call', [call]) for every status.
 *    So stock Baileys can: SEE inbound call events, REJECT them. It carries
 *    ZERO call media. Placing/answering calls needs hand-rolled <call>
 *    stanzas via sock.sendNode() + WhatsApp Web's VoIP WASM stack
 *    (see the earlier spike doc hidden_spike_whatsapp_call.md).
 *
 * 2) werift (pure-TypeScript WebRTC, MIT) — current version 0.25.0
 *    (npm view, 2026-10-09). Zero native deps (debug, buffer, tweetnacl,
 *    @fidm/x509, mediabunny, @noble/curves, multicast-dns, @peculiar/x509,
 *    @shinyoshiaki/binary-data). engines: node >= 22.
 *    Provides: browser-compatible RTCPeerConnection, ICE/DTLS/SCTP,
 *    DataChannel, MediaChannel (sendonly/recvonly/sendrecv, multi-track),
 *    RTP/RTCP, SRTP/SRTCP, MediaRecorder (Opus/WebM).
 *    CAVEAT (from werift docs): "Werift does not implement media-related
 *    features such as codecs" — it moves PACKETS; you bring your own Opus
 *    encoder/decoder. The known-good decoder for WA calls is WhatsApp Web's
 *    own VoIP WASM (its internal Opus variant), not werift.
 *    Honest bottom line: werift CAN carry the RTP/SRTP transport and the
 *    SCTP relay data channel that tunnels UDP to WA's edge relays — pure JS,
 *    so no Termux-native blocker (unlike @roamhq/wrtc). But: (a) this pairing
 *    is UNPROVEN — nobody has shipped werift+WA-VoIP-WASM, so the
 *    pre-negotiated-SCTP relay handshake needs its own mini-spike; (b) it
 *    does NOT decrypt WA's call signaling — that is WhatsApp's own
 *    proprietary protocol, available only as Web's WASM blobs. Bridging WA
 *    signaling into a standard WebRTC peer connection still requires the WA
 *    WASM stack to do the negotiation. No way around that without
 *    reverse-engineering the proprietary crypto.
 *
 * ── What this spike does ─────────────────────────────────────────────────────
 *   Default (dry run): 100% offline. Imports Baileys' REAL exported
 *   signaling parser (getCallStatusFromNode), maps every call-stanza tag to
 *   its event status, asserts the exact API surface of the installed package
 *   by reading its own lib files, and constructs-but-NEVER-SENDS the stanza
 *   shapes an offer/reject would take. No socket, no QR, no network, no JIDs.
 *
 *   Live mode (--live): registers a PASSIVE `wa.ev.on('call')` listener that
 *   only LOGS inbound call events. It NEVER dials, NEVER rejects/answers/
 *   terminates — sending ANY call stanza to a real JID is disabled.
 *   Requires the explicit env flag below (an accident cannot enable it).
 *
 * ── SAFETY (owner's standing rule) ───────────────────────────────────────────
 *   WhatsApp bans/restricts numbers for unofficial-client behavior, and
 *   calls are more conspicuous than texts. The owner's number was already
 *   logged out of Discord by platform enforcement once. DO NOT run --live
 *   on the primary number without the owner's EXPLICIT approval; use a
 *   SECONDARY number. This spike defaults to dry-run for that reason.
 * =============================================================================
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { getCallStatusFromNode } from '@whiskeysockets/baileys';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const has = (f) => args.includes(f);
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};

const LIVE_FLAG = process.env.WA_SPIKE_LIVE === 'I-UNDERSTAND-THE-RISK';
const MODE = has('--live') ? 'live' : 'dry-run';
const PKG = path.join(here, 'node_modules', '@whiskeysockets', 'baileys');

const log = (...a) => console.log(new Date().toISOString(), '[call-spike]', ...a);
const die = (msg, code = 1) => { console.error('[call-spike] FATAL:', msg); process.exit(code); };

// ── Dry run: offline verification against the installed package ──────────────
function verifyBaileysCallSurface() {
  const indexDts = fs.readFileSync(path.join(PKG, 'lib', 'Socket', 'index.d.ts'), 'utf8');
  const callDts = fs.readFileSync(path.join(PKG, 'lib', 'Types', 'Call.d.ts'), 'utf8');
  const msgsRecv = fs.readFileSync(path.join(PKG, 'lib', 'Socket', 'messages-recv.js'), 'utf8');

  const checks = [
    ['rejectCall present in socket API', /rejectCall:\s*\(callId:\s*string,\s*callFrom:\s*string\)/.test(indexDts)],
    ['acceptCall ABSENT from socket API', !/acceptCall/.test(indexDts)],
    ['placeCall ABSENT from socket API', !/placeCall/.test(indexDts)],
    ['hangupCall ABSENT from socket API', !/hangupCall/.test(indexDts)],
    ['WACallEvent type exists', /WACallEvent/.test(callDts)],
    ["statuses include 'offer'", /'offer'/.test(callDts)],
    ["statuses include 'terminate'", /'terminate'/.test(callDts)],
    ["ev.emit('call') in messages-recv.js", /ev\.emit\('call'/.test(msgsRecv)],
    ['rejectCall stanza builder exists', /const rejectCall = async/.test(msgsRecv)],
    ['sendNode exposed (raw stanza hook)', /sendNode/.test(indexDts)],
  ];
  let ok = true;
  for (const [name, passed] of checks) {
    log(`${passed ? 'PASS' : 'FAIL'}  ${name}`);
    if (!passed) ok = false;
  }
  if (!ok) die('Baileys 6.7.24 surface check failed — do not proceed.');
  return { indexDts, callDts };
}

// Map every raw call-stanza tag through Baileys' REAL parser. Offline, no socket.
function demoSignalingParse() {
  log('--- raw <call> tag → WACallUpdateType (via getCallStatusFromNode) ---');
  const cases = [
    { tag: 'offer', attrs: {} },
    { tag: 'offer_notice', attrs: {} },
    { tag: 'ringing', attrs: {} },
    { tag: 'reject', attrs: {} },
    { tag: 'accept', attrs: {} },
    { tag: 'terminate', attrs: { reason: 'timeout' } },
    { tag: 'terminate', attrs: {} },
    { tag: 'relaylatency', attrs: {} },
  ];
  for (const node of cases) {
    const status = getCallStatusFromNode(node);
    log(`  <${node.tag}> attrs=${JSON.stringify(node.attrs)}  →  status='${status}'`);
  }
}

// Construct-but-NEVER-send stanza shapes. These objects are printed only;
// nothing calls query()/sendNode() with them. No JID is ever targeted.
function demoStanzaShapes() {
  log('--- stanza shape: what rejectCall() SENDS (constructed, NOT sent) ---');
  const myJid = '<SELF_JID_PLACEHOLDER>';
  const callFrom = '<CALLER_JID_PLACEHOLDER>';
  const callId = '<CALL_ID_PLACEHOLDER>';
  const rejectStanza = {
    tag: 'call',
    attrs: { from: myJid, to: callFrom },
    content: [{
      tag: 'reject',
      attrs: { 'call-id': callId, 'call-creator': callFrom, count: '0' },
      content: undefined,
    }],
  };
  console.log(JSON.stringify(rejectStanza, null, 2));

  log('--- stanza shape: what a HAND-ROLLED offer would need (constructed, NOT sent) ---');
  log('NOTE: stock Baileys has no placeCall — this would go via sock.sendNode()');
  log('only after the VoIP WASM stack negotiates the call (see findings header).');
  const offerShape = {
    tag: 'call',
    attrs: { from: myJid, to: '<TARGET_JID_PLACEHOLDER — NEVER DIALED>' },
    content: [{
      tag: 'offer',
      attrs: { 'call-id': '<NEW_CALL_ID>', 'call-creator': myJid },
      content: [
        { tag: 'audio', attrs: {}, content: undefined },
        // + <relay> (edge relay list + tokens from WA), <net>, <encopt>,
        //   <dhash>, <capability> — all populated by WhatsApp Web's WASM,
        //   NOT by hand. Building these manually = reimplementing WA's
        //   proprietary signaling. That is the open research problem.
      ],
    }],
  };
  console.log(JSON.stringify(offerShape, null, 2));

  log('--- WACallEvent shape delivered by wa.ev.on(\'call\') ---');
  const sampleEvent = {
    chatId: '<CALLER_JID_PLACEHOLDER>',
    from: '<CALLER_JID_PLACEHOLDER>',
    id: '<CALL_ID_PLACEHOLDER>',
    date: new Date().toISOString(),
    status: 'offer',
    offline: false,
    isVideo: false,
    isGroup: false,
  };
  console.log(JSON.stringify(sampleEvent, null, 2));
}

// ── Live mode: passive listener only ─────────────────────────────────────────
async function runLive() {
  if (!LIVE_FLAG) {
    die([
      'LIVE MODE REFUSED.',
      'To run live you MUST set the env flag explicitly:',
      '  WA_SPIKE_LIVE=I-UNDERSTAND-THE-RISK node bridge/call-spike.mjs --live',
      '',
      '!!! BAN RISK !!! WhatsApp can restrict/ban numbers for unofficial-',
      'client behavior; calls are MORE conspicuous than texts. Never run this',
      'on the primary number without the owner\'s EXPLICIT approval — use a',
      'SECONDARY number. Live mode only LOGS inbound call events; it will',
      'never dial, answer, reject, or terminate a call.',
    ].join('\n'), 2);
  }

  const credsDir = path.resolve(here, opt('--creds', process.env.WA_SPIKE_CREDS || './.call-spike-creds'));
  const bridgeDefaultCreds = path.resolve(here, './.creds');
  if (credsDir === bridgeDefaultCreds) {
    die(`Refusing: --creds points at the main bridge's session (${bridgeDefaultCreds}). ` +
      'A second socket on the same creds would log the bridge OUT. Use a separate creds dir.');
  }

  console.warn('\n' + '!'.repeat(70));
  console.warn('WARNING: live WhatsApp call-event observation is running.');
  console.warn('Personal-account automation risk: WhatsApp may restrict or ban the number.');
  console.warn('This spike NEVER dials and NEVER sends call signaling. Passive log only.');
  console.warn('!'.repeat(70) + '\n');

  const { default: makeWASocket, useMultiFileAuthState, DisconnectReason } =
    await import('@whiskeysockets/baileys');
  const { state, saveCreds } = await useMultiFileAuthState(credsDir);
  const wa = makeWASocket({ auth: state, browser: ['CallSpike', 'Research', '1.0'] });
  wa.ev.on('creds.update', saveCreds);

  wa.ev.on('connection.update', (u) => {
    if (u.qr) log('Scan this QR with WhatsApp > Linked Devices to pair the spike session:');
    if (u.qr) console.log(u.qr);
    if (u.connection === 'open') log('connected — passively listening for `call` events. Ctrl+C to stop.');
    if (u.connection === 'close') {
      const loggedOut = u.lastDisconnect?.error?.output?.statusCode === DisconnectReason.loggedOut;
      log(`connection closed (loggedOut=${loggedOut})`);
      if (loggedOut) die('Logged out — delete creds dir and re-pair.');
      process.exit(0);
    }
  });

  // PASSIVE ONLY: log the exact event shape. No accept/reject/terminate here —
  // sending any of those to a real JID needs owner approval on a 2nd number.
  wa.ev.on('call', (events) => {
    for (const ev of events) {
      log('INBOUND CALL EVENT (passive, nothing sent):', JSON.stringify(ev));
    }
  });

  process.on('SIGINT', () => { log('shutting down (no calls were placed, none ever will be by this spike).'); process.exit(0); });
}

// ── entry ────────────────────────────────────────────────────────────────────
function main() {
  if (has('--help') || has('-h')) {
    console.log(`Usage: node bridge/call-spike.mjs [--dry-run] [--live] [--creds <dir>]`);
    console.log(`  default: --dry-run (offline, safe). --live needs WA_SPIKE_LIVE=I-UNDERSTAND-THE-RISK.`);
    process.exit(0);
  }
  if (MODE === 'live') { runLive().catch((e) => die(`live mode failed: ${e?.message || e}`)); return; }

  log('=== WhatsApp call signaling spike — DRY RUN (offline, no network, no JIDs) ===');
  try {
    log('--- step 1: verify Baileys 6.7.24 call API surface from its own lib files ---');
    verifyBaileysCallSurface();
    log('--- step 2: map raw <call> stanza tags through Baileys\' real parser ---');
    demoSignalingParse();
    log('--- step 3: construct-but-NEVER-send stanza shapes ---');
    demoStanzaShapes();
  } catch (e) {
    die(`dry run failed: ${e?.message || e}`);
  }

  log('=== VERDICT ===');
  log('Stock Baileys 6.7.24: can OBSERVE (ev.on(\'call\')) and REJECT calls. Cannot place or answer.');
  log('Audio path: needs WhatsApp Web VoIP WASM + a relay transport. werift 0.25.0 is the');
  log('pure-JS WebRTC candidate for the transport (no native deps), but moves packets only —');
  log('Opus encode/decode comes from WA\'s WASM, and the werift+WASM pairing is UNPROVEN.');
  log('Open research: offer/accept stanza contents from the WASM stack; relay handshake via werift.');
  log('Live verification needs owner approval on a SECONDARY number. No calls were placed.');
  log('dry run complete — exit 0, zero network side effects.');
}

main();
