# SPIKE: WhatsApp personal-account calling (Baileys) — findings

Date: 2026-10-08. Investigate-only, no production code written, nothing committed.
Status of the thing being spiked: bridge has NO call support today
(`bridge/whatsapp-bridge.mjs` handles text/media/presence/history only).

## Verdict: FEASIBLE WITH CAVEATS

Full-duplex voice calls from the personal WhatsApp account are achievable
in principle — the community has proven the audio path — but three caveats
decide the build: (1) stock Baileys cannot place or answer calls, only
reject; (2) the audio path needs WhatsApp Web's VoIP WASM stack plus a
WebRTC relay transport, and the known transport (`@roamhq/wrtc`) is
native/glibc-only → **blocked on Termux**; (3) inbound answering is only
proven in forks, not the base project.

## 1. Baileys call API surface (verified against installed 6.7.24)

From `bridge/node_modules/@whiskeysockets/baileys/lib/Types/Call.d.ts`:

```ts
export type WACallUpdateType = 'offer' | 'ringing' | 'timeout' | 'reject' | 'accept' | 'terminate';
export type WACallEvent = {
    chatId: string; from: string; isGroup?: boolean; groupJid?: string;
    id: string; date: Date; isVideo?: boolean; status: WACallUpdateType;
    offline: boolean; latencyMs?: number;
};
```

- Events: `wa.ev.on('call', (events: WACallEvent[]) => ...)` — inbound offer,
  ringing, accept, reject, terminate, timeout. Present in 6.7.24. (`lib/Socket/messages-recv.js`
  parses the raw `<call>` stanzas; offers are cached in `callOfferCache`.)
- Actions in stock Baileys: **only** `rejectCall(callId, callFrom)`
  (`lib/Socket/index.d.ts:49`). There is NO acceptCall / placeCall in
  6.7.24. Signaling-only; Baileys carries zero call media.

So stock Baileys = call *signaling observer* (see offers, reject them).
Everything else needs the VoIP stack below.

## 2. The audio path — how the community solved it (2025–2026 state)

WhatsApp calls are: Baileys signaling + a separate media plane
(Opus/RTP/SRTP to WhatsApp edge relay servers). The working approach,
proven by [baileys-caller](https://github.com/sheiitear/baileys-caller) (MIT):

1. **Signaling** — Baileys for auth + `<call>` stanzas.
2. **VoIP WASM stack** — WhatsApp Web's official VoIP WASM
   (`whatsapp.wasm` + loader + worker modules, refreshable via
   `npm run fetch-wasm`) runs in-process in Node: negotiates the call,
   encodes/decodes Opus, manages the RTP/SRTP session. Loaded in a
   `vm.Context` with a `worker_threads` pthread-pool mirror (20 workers
   upstream; a fork cut it to 4 and measured ~63% RSS reduction).
3. **Relay transport** — tunnels UDP to WhatsApp's edge relay servers via
   **WebRTC data channels (pre-negotiated SCTP, custom DTLS fingerprint)**
   using `@roamhq/wrtc`. Relay list/tokens come from the call signaling.

Proven status (upstream `baileys-caller`):
- ✅ outbound 1:1 voice calls, ✅ stream MP3/WAV audio in,
  ✅ remote audio out as **16 kHz mono `Float32Array`**, ✅ mute/unmute/hangup
- ❌ group calls, ❌ video, ❌ **inbound calls**

Inbound answering exists only in forks:
- [chama-baileys-caller](https://github.com/chamanemax02/chama-baileys-caller/blob/HEAD/README.md)
  — `acceptCall` auto-answer + audio streaming + auto-hangup.
- [japofc/baileys](https://github.com/japofc/baileys) — bundled WASM call
  stack + `attachVoip(sock, { autoAnswer: true })`, `answerCall(callId)`,
  group-call join.
- [lreviza hoshino-voip](https://github.com/revizahoshii-no/lreviza-baileys/blob/HEAD/packages/voip/README.md)
  — uses `libmlow-wasm` (WhatsApp's Opus variant), RTP/SRTP, STUN,
  WebRTC/SCTP relay; "`@roamhq/wrtc` — for real calls".

**Bottom line on audio:** full-duplex PCM relay is real and demonstrated.
Outbound is the safe bet; inbound = take a fork's accept path or
reimplement the `<call>` accept stanza + WASM accept flow ourselves.

## 3. How it wires into Devon's voice stack

The fit is unusually clean — no new audio science needed:

- `nomorals/voice/session.py` works in **16 kHz mono int16 PCM, 30 ms
  chunks** (`SAMPLE_RATE=16000`, `CHUNK_SAMPLES=480`). baileys-caller emits
  16 kHz mono `Float32Array` — conversion is a multiply-and-cast.
- `VoiceSession` takes injected `MicSource` / `SpeakerSink` protocols and
  is fully testable without hardware. A WhatsApp call is just another
  mic/speaker pair:
  - `WACallMic.read_chunk()` ← bridge `call_audio` events (base64 PCM)
  - `WACallSpeaker.play(wav)` → resample to 16 kHz mono → bridge
    `call_audio` command → WASM → relay
- Reuse for free: EnergyVAD turn-taking, barge-in, `speakify()` shaping,
  Groq Whisper STT (`make_bridge_stt` / `make_local_stt`), UniversalTTS
  (XTTS-private on workstation, Edge/Chatterbox elsewhere), consent +
  stats stores. The Telegram voice-note loop (`pingpong.py`) proves the
  STT→think→TTS chain; the live `VoiceSession` proves the real-time loop.
- Bonus: a call needs **no phone mic/speaker** — Termux never touches
  `sounddevice`. The call *is* the audio device.

### Minimal viable wiring

```
WhatsApp relay --SRTP/Opus--> VoIP WASM (Node, in bridge process)
        --> 'audio' Float32Array 16k mono --> base64 JSON-lines -->
Python WhatsAppAdapter --> WACallMic (MicSource) --> VoiceSession
        --> think (PartnerRuntime.handle_message, same as chat)
        --> TTS wav --> WACallSpeaker --> base64 JSON-lines -->
bridge 'call_audio' cmd --> WASM --> relay --> caller hears Devon
```

Call control rides the same TCP socket as new JSON-lines commands:
`call_place`, `call_answer`, `call_reject`, `call_hangup`,
`call_audio` (both directions), and bridge→python `call_event`
(offer/ringing/connected/ended) + `call_audio` frames.

Audio bandwidth over localhost: 30 ms of 16 kHz mono int16 = 960 bytes
(~1.3 KB base64 per line). Trivial.

## 4. Termux / Android feasibility

| Piece | Termux verdict |
|---|---|
| Node | ✅ Need ≥ 20; Termux `nodejs` package is current (22/24). |
| Baileys 6.7.24 | ✅ Pure JS, already runs on the phone. |
| VoIP WASM | ⚠️ Likely OK — V8 on Node/aarch64 runs WASM incl. shared-memory threads; `worker_threads` work on Termux. 20-thread pool is heavy → take the 4-worker cut from the Starsky fork. Needs a live test. |
| `@roamhq/wrtc` | ❌ **Hard blocker.** Native libwebrtc addon; ships glibc-only prebuilts, no musl, source build infeasible — Termux is bionic libc, worse. Will not install. |
| Relay transport w/o wrtc | ⚠️ Open question. `werift` is pure-TypeScript WebRTC (zero native, SCTP data channels supported) and could replace wrtc for the relay data channel — **unproven, needs its own mini-spike.** `node-datachannel` is also native (probably bionic-blocked). |
| ffmpeg | ✅ Needed for decoding streamed audio files (Termux has an `ffmpeg` package). For live TTS PCM we bypass ffmpeg entirely. |

**Single biggest risk:** the relay transport on Termux. Everything else
has a known-good answer.

## 5. Concrete build plan (if green-lit)

1. **`bridge/package.json`** — add `baileys-caller` via git URL (or vendor
   the ~6 source files: `wasm-engine`, `relay-transport`, `signaling`,
   `audio-feeder`, `types`, `index`). Pin `@whiskeysockets/baileys` compat
   (upstream wants `^7.0.0-rc11`; our 6.7.24 has the `call` events — verify
   the signaling hooks work against 6.7.24 or bump).
2. **`bridge/whatsapp-bridge.mjs`** — new `voip` section (lazy-loaded so
   the bridge still runs without call deps):
   - `wa.ev.on('call')` → broadcast `{type:'call_event', event, call_id, from, is_video}`.
   - commands: `call_place {chat}`, `call_answer {call_id}`,
     `call_reject {call_id}`, `call_hangup {call_id}`,
     `call_audio {call_id, pcm_b64}` (outbound PCM → WASM).
   - WASM `onAudioPlaybackData` → broadcast `{type:'call_audio', call_id, pcm_b64}`.
   - Inbound: port the accept path from chama-baileys-caller or japofc fork
     (offer → accept stanza + WASM accept) — do NOT ship auto-answer
     default-on; owner opts in per call.
3. **Termux transport spike (first!)** — try `werift` as the
   `RelayRtcTransport` WebRTC provider on Termux Node; fall back to
   documenting "calls work on workstation/laptop profiles only"
   (profile-gate, per standing rule — never design the capability down).
4. **`nomorals/social/chat/whatsapp_call.py`** (new) — `WACallMic`,
   `WACallSpeaker` implementing `session.py` protocols over the bridge
   protocol; `WhatsAppCallSession` that builds a `VoiceSession` with
   `think=PartnerRuntime.handle_message`, per-call consent check, and
   `deliver_text` → WhatsApp chat (voice summary spoken, full text to chat —
   same pattern as `speakify`).
5. **`nomorals/social/chat/whatsapp.py`** — `WhatsAppAdapter` gains
   `call_place/call_answer/call_reject/call_hangup/send_call_audio`
   (thin `_send_cmd` wrappers) and dispatches `call_event`/`call_audio`
   to the gateway/handler; `health()` reports call capability.
6. **Tests** — protocol round-trip for the new commands (offline, mocked
   socket), Float32→int16 conversion, VAD over call-shaped audio; no live
   call in CI.
7. **Live validation** (needs the user + a second device): place call →
   confirm audio heard both ways → inbound answer → full conversational
   loop → hangup. Same bar as the Telegram voice-note verification.

## 6. Account-risk note (standing order: warn upfront)

Like the Discord token, a linked-device VoIP session is automation on a
personal account. WhatsApp can and does ban numbers for unofficial-client
behavior; calls are *more* conspicuous than texts. Recommend: test with a
secondary number first if the user has one; keep call features
owner-initiated or explicitly opted-in, never auto-answering strangers.

## Open questions for the build

- Does baileys-caller's signaling layer work against our Baileys 6.7.24,
  or must we bump to 7.x-rc (and does 7.x-rc break the text bridge)?
- Can `werift` stand in for `@roamhq/wrtc` in `RelayRtcTransport`
  (pre-negotiated SCTP data channel + custom DTLS fingerprint)?
- Does the VoIP WASM run stably under Termux Node (memory/CPU on a phone
  during a live call)?
- Inbound: lift accept logic from a fork vs. write our own accept stanza —
  decide after reading chama-baileys-caller's `acceptCall` implementation.
