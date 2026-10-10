# VOICE SWEEP — External Mining Report

Mined 2026-10-10. Every significant class in `nomorals/voice/` compared
against the best implementations found OUTSIDE the repo (GitHub repos,
model cards, engineering write-ups — including weak/"trash" builds, per
the standing order). Gold adopted below; nothing invented.

---

## 1. `UniversalTTS` + backends (tts.py)

**Current:** 13 backends (bark, xtts, kokoro, cosyvoice, dia, orpheus,
hf-endpoint, chatterbox, piper, f5tts, omnivoice, qwen3tts, system +
lazy diffsinger), audience licensing (XTTS private-only), quality/latency
scoring, sentence-chunked streaming fallback, segment cache, native-tag
rendering per backend, emotion-DSP fallback for tag-less backends.

**Best outside:**
- `ngocthanhluong/tts-audio-suite` (ComfyUI): multi-engine hub with
  per-character voice switching (`[CharacterName]`), per-character voice
  inheritance, SRT processing, parameter-based cache invalidation.
- `KittenML/KittenTTS` (Apache-2.0): 15–80M param ONNX, 25MB smallest,
  CPU-only, 8 built-in voices, `--speed`, 24kHz, SSE streaming server
  (`acovo/kitten_tts_rs`).
- `SparkAudio/Spark-TTS` (Apache-2.0): 0.5B, ~50x realtime, zero-shot
  from prompt text+audio, controllable gender/pitch/speed,
  `python -m cli.inference` interface, Triton serving reference.
- `Zyphra/Zonos` (Apache-2.0): 1.6B, **8-D emotion vector**
  (happiness/sadness/disgust/fear/surprise/anger/other/neutral),
  speaking-rate/pitch/frequency conditioning, 44.1kHz, audio-prefix
  inputs (whisper elicitation).
- `FunAudioLLM/CosyVoice 3` (Apache-2.0): instruction control
  (emotion/style/dialect), ~150ms streaming, fastest emotion-capable.
- `ResembleAI/Chatterbox` (MIT): blind-test winner vs ElevenLabs,
  `generate_stream` chunked API, exaggeration dial.
- `Step-Audio-EditX` (Apache-2.0): edit speech via natural-language
  instruction (the DSP-fallback idea, but neural).

**Gold taken:**
- New backends: **KittenTTS** (the missing tiny-CPU streaming tier —
  nothing in the module runs well on bare CPU with cloning-adjacent
  quality), **Spark-TTS** (fastest free zero-shot, controllable
  gender/pitch/speed — new control surface), **Zonos** (the 8-D emotion
  vector is the most controllable emotion interface in open TTS; maps
  1:1 onto the director's canonical emotions).
- `speak_batch()`: parallel multi-text synthesis (longform chapters,
  dialogue turns) — every production hub does this; we did serial.
- Per-call `speed`/`exaggeration` overrides plumbed to backends that
  support them (Kitten `--speed`, Chatterbox exaggeration).
- `backend_info()`: one dict of capabilities/license/needs per backend
  (the efficiency RESOURCE_BUDGETS lived in a different file from the
  capability table — merged view).

**Gaps closed:** no sub-100MB CPU streaming backend; no batch API; no
emotion-vector backend; Spark/Zonos install probes.

## 2. `UniversalSTT` + backends (stt.py)

**Current:** faster-whisper, Parakeet (onnx_asr), whisper.cpp,
classic whisper, HF endpoint. Clean cascade.

**Best outside:**
- `SYSTRAN/faster-whisper`: `word_timestamps=True`, `vad_filter=True`
  with `vad_parameters=dict(min_silence_duration_ms=500)` — Silero VAD
  built in, conservative 2s default.
- `KoljaB/RealtimeSTT`: streaming faster-whisper, the live-loop STT
  pattern.
- `WhisperX`: word-level alignment + pyannote diarization.
- LocalAI `parakeet-cpp`: streaming cache-aware, `realtime_eou`
  end-of-utterance events, companion diarization/classification models.
- forasoft (2026): diarization done right = cluster on the fly, map
  speaker IDs→names server-side, 400–600ms hysteresis on switches.

**Gold taken:**
- `transcribe(..., word_timestamps=..., vad_filter=...)`: kwargs passed
  through to faster-whisper (were silently dropped before).
- `to_srt()` / `to_vtt()`: subtitle export from segments (tts-audio-suite
  has SRT; we had none).
- `StreamingSTT`: chunked incremental transcription with partial
  callbacks — the RealtimeSTT pattern, minus the dependency.
- `diarize()`: optional pyannote hook with hysteresis + server-side
  speaker-name mapping; honest "not installed" when absent.

## 3. `VoiceSession` / `EnergyVAD` (session.py)

**Current:** full-duplex loop, consent store, stats, barge-in, speakify,
room calibration.

**Best outside:**
- Pipecat voice agents: `SileroVADAnalyzer` on the transport;
  **barge-in cancels LLM+TTS in flight and stops playback**; history
  records what was *actually played*, not what was generated
  (interrupted sentences must not enter history as complete).
- `kanak234/vaani` TRD: Silero VAD **ONNX** (1.8MB, <1ms/32ms frame on
  CPU) over the torch build (~2.5GB for a 1.8MB model); energy VAD kept
  as zero-dependency fallback/diagnostics.
- faster-whisper `vad_parameters`: `min_silence_duration_ms=500` keeps
  natural cadence.

**Gold taken:**
- `SileroVAD` class: ONNX-runtime Silero, drop-in for `EnergyVAD`
  (`observe()`/`is_speech()` protocol-compatible), graceful fallback.
- `VoiceSession.interrupt()`: public barge-in API (was inline only).
- `spoken_history`: the session now tracks what audio was *actually
  played* per turn (Pipecat lesson) and `SessionReport.format()` renders
  a god-tier session card (turns, barge-ins, latency p50/p95, per-turn
  rows) instead of a bare dataclass repr.

## 4. `PerformanceTuner` / director renderers (director.py)

**Current:** 30 bursts, 53 emotions, intent detection, per-backend
native renderers (fish/dia/orpheus/chatterbox/omnivoice/cosyvoice/
bark/qwen3/plain), EFFECT_PRESETS.

**Best outside:**
- ElevenLabs v3: audio tags; Hume Octave: NL emotion instruction
  ("sound sarcastic"); Zonos: 8-D emotion vector; Dia: 21 real
  audio-event tags; CosyVoice 3: instruct tokens.

**Gold taken:**
- Renderer registry (`_RENDERERS` dict + `supported_backends()`):
  `render_for` was an if-chain; now data-driven and extensible.
- New renderers: `render_zonos` (canonical emotion → 8-D vector tag),
  `render_kitten` (plain + speed-aware), `render_spark` (control tags).
- `STYLE_PRESETS` (audiobook narrator, podcast duo, announcement,
  bedtime story, hype): preset → (mood, intensity, effect, pace) —
  the "house style" one-liner the module lacked.

## 5. `DiffSingerBackend` / singing.py

**Current:** melody parsing, f0 curve with portamento + vibrato,
DiffSinger voicebank inference, RVC timbre swap.

**Best outside:**
- `KakaruHayate/DiffSinger` (OpenVPI): expression curves —
  PITD (pitch), DYN (dynamics), GENC (gender/formant), VELC
  (velocity), ENE (energy), BREC (breathiness), TENC (tension),
  VOIC (voicing), PEXP (pitch expressiveness), SHFC (tone shift).
- `Soul-AILab/SoulX-Singer` (Apache-2.0): zero-shot singing voice
  synthesis, F0-contour + MIDI-note conditioning.

**Gold taken:**
- Per-note expression: `dynamics` (DYN), `breathiness` (BREC),
  `tension` (TENC) fields on `Note`; `humanize()` (timing/pitch jitter —
  the anti-robot pass); `export_midi()` (notes → SMF for DAWs);
  extended `parse_melody` (`vib:`, `dyn:` inline modifiers).

## 6. ambience.py (`generate_ambience`, `Room`)

**Current:** rain/wind/thunder/crowd/applause, Room reverb presets,
mix_under, with_ambience.

**Best outside:** foley/scene design = layered beds (not single noise
sources); scene described in words → layers.

**Gold taken:** new beds (cafe, fireplace, night crickets, ocean,
rain-on-window), `AmbienceScene` (named layers + per-layer levels +
fades), `describe_scene("rainy cafe at night")` keyword parser.

## 7. mastering.py

**Current:** dc → deess → normalize → soft-limit chain.

**Best outside:** dhwani pipeline (dynamic EQ, multiband compressor,
LUFS normalization, true-peak limiter).

**Gold taken (honest pure-Python subset):** `trim_silence`,
`fade_in_out`, `loudness_match` (RMS-based, LUFS-inspired target),
`compress` (feedforward compressor with attack/release),
`master()` gains `loudness`/`trim`/`fade` options and reports stages.

## 8. emotion_dsp.py

**Current:** pitch_shift, time_stretch, energy, breathiness,
shape_emotion.

**Best outside:** Step-Audio-EditX (instruction → speech edit).

**Gold taken:** `whisperize` (whisper = breathiness + HF noise +
energy dip, one call), `tremolo`, `vibrato_dsp`, `apply_delivery`
(whisper/shout presets incl. delivery verbs from nl_director).

## 9. biometrics.py

**Current:** acoustic voice-print enroll/verify, conservative
thresholds.

**Best outside:**
- `speechbrain/spkrec-ecapa-voxceleb`: ECAPA-TDNN, EER 0.80% —
  the open standard for speaker verification.
- `resemble-ai/Resemblyzer`: 5–30s enrollment, similarity metric.
- `anderson-venture/voice-indentification` pipeline (the best small
  design found): resample → Silero VAD → reject <2s speech →
  countermeasure FIRST (40x cheaper, short-circuits) → embedder →
  AS-Norm → cosine → `decide()` with FOUR outcomes
  (IDENTIFIED/UNKNOWN/AMBIGUOUS/REJECTED); "Never feed non-speech to
  the embedder"; "Returning UNKNOWN is a success, bias hard toward it."

**Gold taken:** multi-sample enrollment (mean print — Resemblyzer
pattern), `decide()` 4-outcome API (IDENTIFIED/UNKNOWN/AMBIGUOUS/
REJECTED) replacing the bool|None, `liveness_challenge()` (speak-back
random digits — anti-replay), `spoof_score()` heuristic (flat
pitch/energy variance — cheap countermeasure-first ordering).

## 10. catalogue.py (`VoiceCatalogue`)

**Current:** add/remove/get/list, active voice, per-chat voices,
clone, speak/speak_as.

**Best outside:** tts-audio-suite character switching + per-character
voice inheritance.

**Gold taken:** `search()` (name/tag/backend), `rename()`,
`format_table()` (god-tier voice list card with backend, profile,
usage), `stats()` usage counters recorded on speak.

## 11. design.py (`VoiceDesign`, `shape_voice`, `morph_voices`)

**Current:** describe→params, tone shaping, 2-voice morph.

**Best outside:** OpenVoice tone-color/style separation (already the
basis of accent.py).

**Gold taken:** `blend_voices()` — N-voice weighted blend (morph was
2-voice only); `describe` gains intensity adverbs.

## 12. longform.py (`LongFormSynthesizer`)

**Current:** sentence chunking, rolling prosody, crossfade, degenerate
detection + regen.

**Best outside:** audiobook pipelines (chapter marks, SRT timing,
resume).

**Gold taken:** `progress_cb` per chunk, `chapters` output (title,
start_s, end_s), `resume_from` chapter index.

## 13. efficiency.py (`SegmentCache`, `LatencyTable`)

**Current:** LRU + disk cache, latency table, resource budgets.

**Best outside:** tts-audio-suite parameter-based cache invalidation;
RealtimeSTT model caching.

**Gold taken:** `invalidate()` (prefix/tag invalidation),
`clear()`, `warm()`, TTL on disk entries, `size_report()`.

## 14. fetch.py

**Current:** piper voice + generic HF model fetch.

**Best outside:** `snapshot_download` everywhere; per-model fetchers.

**Gold taken:** `fetch_kitten_model()`, `fetch_spark_model()`,
`fetch_chatterbox_model()`, `fetch_kokoro_model()`, `fetch_zonos_model()`
— one-liners that put weights where the backends look.

## 15. dialogue.py

**Current:** parse/stitch/backend-split detection.

**Best outside:** tts-audio-suite `[CharacterName]` switching with
per-character voice inheritance + pause tags.

**Gold taken:** `render_dialogue()` — full pipeline (parse → voice map
→ per-turn synth → stitch) with per-turn pause from punctuation and
`estimate_duration()`; speaker→voice auto-mapping with pitch-offset
fallback (existing behavior kept, now one call).

## 16. nl_director.py

**Current:** prose direction parsing, DSP param mapping.

**Best outside:** Hume Octave NL instruction; Zonos dimensional
emotion.

**Gold taken:** vocabulary x2 (deliveries: rap, croon, preach…;
emotions: 60+; accents incl. pidgin), intensity adverbs ("very",
"slightly" → level 1–5), `Direction.to_dsp_params()` method,
`describe()` pretty-printer.

## 17. accent.py

**Current:** accent-first hybrid (multilingual render + RVC identity).

**Best outside:** CosyVoice 3 dialect/instruct control.

**Gold taken:** `strength` 0–1 → prosody nudge via emotion_dsp on the
accent-only tier (was pure pass-through); `list_accents()` with hints.

## 18. neural_emotion.py / conversation.py / pingpong.py / bridge.py /
## money.py / rvc_bridge.py

**Best outside:** Pipecat latency meters; Telegram voice-note OGG;
WhatsApp voice notes.

**Gold taken:**
- `render_emotional(..., intensity=)` scaling.
- `conversation.voice_turn` → structured result (transcript, reply,
  timings, voice_used).
- `pingpong.synthesize_voice_reply(..., progress_cb=)` + `voice_note_info()`.
- `bridge.VoiceBridge.broadcast_voice()` (multi-chat fan-out with
  per-chat summary) + delivery receipts.
- `money.format_confirmation()` — styled readback ("Sending ₦5,000 to
  Ada — say YES to confirm").
- `rvc_bridge.model_info()` (name, path, size, mtime).

## 19. Presentation / style layer

Voice is heard, not seen — but every surface the user *reads* (voice
lists, session reports, confirmations, progress) was bare. Added:
`catalogue.format_table()`, `SessionReport.format()`,
`money.format_confirmation()`, `longform` progress rows,
`describe_scene` echo. God-tier, not functional.
