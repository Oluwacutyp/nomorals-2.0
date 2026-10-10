# TTS Mining Report — God-Tier Voice

Written before any build commits. Surveyed: ElevenLabs v3/v4, Fish Audio, Dia, Chatterbox, CosyVoice 3, Zonos, Step-Audio-EditX, IndexTTS2, Cartesia, Hume, CSM, Bark, Orpheus, plus trash-build discipline lessons.

## 1. ElevenLabs v4 (launched ~Sept 2026)

**Gold:**
- **Natural language direction INSTEAD of rigid markup**: `[said angrily in British accent]`, `[light rain]`, `[phone buzzing]` — SSML break tags DISABLED in favor of inline descriptors. The model parses intent from prose.
- **Environmental audio inline**: `[light rain]`, `[door slams]` — sound effects at script-level timing, no separate production step.
- **Multi-speaker with scene context**: understands the whole conversation, speakers respond to what was just said — not stitched isolated lines.
- **Speaker consistency**: new identity-capture method keeps voice stable across generations (audiobooks, long conversations).
- **v4 Turbo**: ~100ms to first response — faster than human conversational gap. Expressive AND real-time.
- **Expanded IPA support**: phoneme-level pronunciation for medical terms, proper nouns, localized terminology.

**Take:** NL direction parsing (not just tag matching). Environmental audio tags. Speaker-consistency tracking. 100ms first-chunk as the bar.

## 2. Open Models (2026 landscape)

### Dia-1.6B (Nari Labs, Apache-2.0)
- **21 real audio-event tags** (laughs/cries/gasps) — not faked, actually in training data
- **Dialogue-first**: two-speaker scripts with `[S1]`/`[S2]` prefixes, native turn-taking
- RTF 0.77 compiled, 6.2GB VRAM — fits alongside an LLM on 12GB
- **Take:** the multi-speaker dialogue architecture. Already in our registry.

### Chatterbox family (Resemble AI, MIT)
- **Turbo**: native `[laugh]`/`[chuckle]`/`[cough]`, sub-200ms latency, BEAT ElevenLabs in blind test (65% vs 25%)
- **Nano**: 110M params, **3x realtime on 8 CPU cores** — the phone path
- **Multilingual V3**: 23 languages, exaggeration dial
- **Take:** MIT license = use freely. Nano is the efficiency king. Already our primary.

### CosyVoice 3 (FunAudioLLM, Apache-2.0)
- **Instruction control**: natural-language prefix ending in `<|endofprompt|>` — "speak very happily" — 100+ instruction types claimed
- 0.5B, 4.5GB VRAM, streaming ~150ms TTFA
- **Take:** NL instruction pattern validates the v4 direction. Instruct > tags for open models.

### Zonos-v0.1 (Apache-2.0)
- **8-dimensional emotion vector** — the most explicitly controllable open model
- **Take:** dimensional emotion control (not categorical). Steal the 8-D concept for our tag→parameter mapping.

### Step-Audio-EditX (stepfun-ai, 3B, Apache-2.0, Nov 2025)
- **Emotion EDITING of existing audio** via bracket tags `[Angry]`, `[Whispering]`
- 14 emotions, 30+ styles, paralinguistics — 83.4% emotion-edit accuracy
- **KEY INSIGHT**: converts ANY neutral TTS output into emotional versions with identical content/speaker
- **Take:** THIS is how tags WORK on backends without native support. Don't strip, don't fake with onomatopoeia — generate neutral, then EDIT the emotion in. This is the missing piece.

### Fish Audio S2
- 15,000+ free-form `[tag]` directions — near pass-through
- **Caveat**: S2 Pro needs commercial license; S1-mini is the open one
- **Take:** free-form tag philosophy (don't limit vocabulary).

### IndexTTS2
- **Disentangled emotion vs speaker** — independent control
- SOTA WER/sim/emotion but 14GB — won't fit alongside LLM
- **Take:** the disentanglement principle for our voice design.

### Cartesia Sonic-3 (cloud)
- **TTFA 40-90ms** — fastest measured. Nonverbal tags + inflection.
- **Take:** 40-90ms is the cloud bar. Our local bar: <300ms.

### Hume Octave (cloud)
- Auto emotional context + NL instructions, LLM trained jointly on text+speech+emotion
- **Take:** joint training is the future; for now, emulate with pipeline.

### CSM-1B (Sesame, Apache-2.0)
- Conversational, multi-turn audio context
- **Take:** dialogue context architecture.

## 3. Efficiency Landscape

| Backend | TTFA/Latency | RTF | Size | Phone-viable |
|---|---|---|---|---|
| Chatterbox Nano | ~200ms | 0.33 (3x RT) | 110M | YES (8 CPU cores) |
| Chatterbox Turbo | sub-200ms | 0.50 | 500M | GPU preferred |
| CosyVoice 3 | ~150ms stream | 0.52-0.63 | 0.5B | Tight |
| Dia-1.6B | — | 0.77 | 1.6B/6.2GB | No (GPU only) |
| Piper | sub-300ms | fast | tiny | YES |
| Kokoro-82M | sub-300ms | fast | 82M | YES |
| Cartesia (cloud) | 40-90ms | — | — | N/A |
| ElevenLabs Flash | ~75ms | — | — | N/A |

**Take:** Nano + Piper + Kokoro are the phone stack. Turbo for quality+speed. Dia for dialogue (GPU).

## 4. Trash-Build Discipline Lessons

- Verify tags against actual audio output, not just parser tests — a tag that parses but doesn't change the waveform is a lie
- Don't ship 60+ tags when 12 actually render — audit the RENDERED set
- Cache at the segment level, not the utterance level — repeated phrases across different utterances still hit
- Measure TTFA per backend on first run, store it, route by it

## 5. What We're Building (from this mining)

1. **NL direction parser** (v4-style): `[said angrily in British accent]` → structured direction, not just tag lookup
2. **Emotion editing pipeline** (Step-Audio-EditX pattern): neutral → emotional via post-process. Tags WORK everywhere.
3. **Multi-speaker dialogue**: Dia-style `[S1]`/`[S2]` with turn-taking and prosody matching
4. **8-D emotion space** (Zonos-inspired): dimensional, not just categorical
5. **Efficiency core**: segment-level cache, per-backend latency table, quantized defaults, streaming-first routing
6. **Accent/dialect**: Nigerian English, Yoruba, Ekiti — via instruction-tuned backends + phoneme hints
7. **Environmental audio**: `[light rain]` etc. — mixed in post (honest: not generated by TTS model)
