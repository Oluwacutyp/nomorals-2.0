# VOICE MINING REPORT — Phase 2: Voice (god-tier upgrade)

Mined 2026-10-10. Inside: all of `nomorals/voice/` (16 modules, ~8300 lines),
`runtime_voice.py`, spine `tools/audio.py`. Outside: DiffSinger, RVC, StyleTTS 2,
OpenVoice, long-form TTS research (LACI), procedural ambience literature.

## What's already real (keep, don't fork)

- **UniversalTTS** (`tts.py`, 2645 lines): 13 backends (Chatterbox, F5, OmniVoice,
  Qwen3, Bark, XTTS, Kokoro, Piper, CosyVoice, Orpheus, Dia, HF-endpoint, system),
  audience-licensed (XTTS private-only), purpose-scored selection, tag processor,
  director with 30 vocal bursts / 53 emotions, voice profiles + clone audit.
- **UniversalSTT**: faster-whisper → parakeet → whisper.cpp → openai-whisper → HF.
- **VoiceSession** (duplex loop): energy VAD, barge-in, one capture thread, AES
  keep-audio. **pingpong**: voice-note turns. **conversation**: voice chat mode.
  **biometrics**: owner voice-print gating. **bridge**: TTS→WhatsApp/Telegram.
  **design**: voice morphing via DSP. **fetch**: HF model registry.
  **efficiency**: segment cache + latency table. **money**: voice payments.

## The 5 named gaps → mined solutions

### 1. True neural emotion (vs DSP pitch/rate hacks)
GOLD — the RVC hybrid pattern (used by voice-studio, dhwani):
RVC is speech→speech conversion that PRESERVES the source's emotion, prosody,
timing. So: synthesize the performance expressively (expressive backend, or
even DSP-shaped), then RVC-convert into the target identity → the emotion
transfers neurally. RVC: MIT, 5–10 min training data, 90–170ms latency,
real-time capable. Also: StyleTTS 2 (style from reference audio),
CosyVoice2 instruction-driven emotion, OpenVoice tone-color/style separation.
**Build:** `neural_emotion.py` — render emotional base → RVC to target voice.
DSP stays as the honest fallback chain.

### 2. Singing (parsed but not wired)
GOLD — DiffSinger: MIT, diffusion-based SVS, conditioned on score
(lyrics + pitch/F0 or MIDI). OpenUtau voicebanks are drop-in files; ONNX
acoustic model + NSF-HiFiGAN vocoder runs on CPU (~2s per 3s vocal, per
songmaker). Then RVC supplies the user's timbre (mitystudio pattern).
**Build:** `singing.py` — DiffSinger backend registered in UniversalTTS,
melody input (MIDI/note list + lyrics), honest unavailable without
voicebank/runtime. DSP "chant" stays as fallback.

### 3. Accent conversion (parsed but decorative)
GOLD — OpenVoice separates tone color (identity) from style (accent lives
in style); hybrid route: synthesize in target accent via multilingual
backends (Chatterbox multilingual, CosyVoice cross-lingual) → RVC identity
transfer keeps the accent. Trash-tier fallback: espeak-ng phonemizer accent
perception hints. **Build:** `accent.py` — accent-first synthesis + RVC,
honest path reporting.

### 4. Ambience mixing (parsed but never mixed)
GOLD — procedural audio literature: rain = bandlimited equalized noise
(diffuse) + chirped/modal noisy impacts (droplets); wind = slow-modulated
filtered noise; room = feedback delay network / Schroeder reverb, all pure
Python on sample arrays. **Build:** `ambience.py` — procedural generators
(rain, wind, crowd, thunder, room tone, phone line), FDN-ish reverb, ducking
mixer. Wire into nl_director's ambient parsing: `[light rain]` finally works.

### 5. Long-form consistency (drifts across chunks)
GOLD — research: context-aware style prediction (hierarchical transformers
over previous utterances); LACI (error detection + rollback + regenerate for
long-form AR TTS); voice-studio's long-form preprocessing (denoise/normalize/
chunk). Practical version: rolling prosody context (previous chunk seeds the
next), 50ms crossfade stitching, degenerate-chunk detection (silence/energy
collapse → regenerate once), one voice anchor for the whole document.
**Build:** `longform.py` — LongFormSynthesizer over UniversalTTS.

## Other gold from the trash
- dhwani: per-sentence dynamic EQ, de-esser, multiband compressor, LUFS
  normalization, true-peak limiter — a mastering chain after synthesis.
  → Add `mastering.py`: gentle, pure-Python (normalize, soft limiter, de-ess).
- voice-studio: multi-character podcast script (`NAME: dialogue`) with smooth
  transitions → extend dialogue.py with NAME: format + crossfade turns.
- mitystudio: engine tier/downgrade guard — honest capability tiers per path.
  → Apply: every new module reports its active tier.

## Build order
1. `rvc_bridge.py` (foundation for emotion/accent/singing identity)
2. `ambience.py` (self-contained, pure DSP)
3. `neural_emotion.py` (on rvc_bridge)
4. `accent.py` (on rvc_bridge)
5. `singing.py` (DiffSinger backend + RVC)
6. `longform.py` (over UniversalTTS)
7. `mastering.py` (post chain)
8. Wire into UniversalTTS.speak/perform + spine tools + fetch registry
9. Tests: `tests/test_voice_godtier.py`
