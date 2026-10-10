# AUDIO SWEEP — External Mining Report

Date: 2026-10-10. Method: real web search for every significant class in
`nomorals/audio/`, best AND trash implementations mined before any code.

Sources consulted are linked per section. Nothing here is invented — every
"the best does X" below comes from an actual implementation found in search.

---

## 1. fingerprint.py — Shazam-style fingerprinting + acoustic analysis

**Best implementations mined:**

- **Wang (2003) Shazam patent pipeline** — audio → Hann-windowed STFT →
  dB-magnitude spectrogram → local-maximum peaks (constellation, via a
  `maximum_filter` neighborhood in freq×time, not band slicing) → pair each
  anchor peak with the next ~15 peaks → hash `(f1, f2, Δt)` + anchor time
  `t1` → hash table `hash → (song, position)` → offset histogram, tallest
  spike wins. Temporal alignment is the load-bearing idea: random collisions
  never form a coherent line.
  (github.com/ebenbaruk/shazam-fingerprint-algorithm,
   github.com/thmsgo18/shazam, github.com/kushn/signals-systems-project)
- **ebenbaruk/shazam-fingerprint-algorithm**: 4096-point FFT, numba, librosa,
  SQLite storage, `add`/`identify` CLI, 5-second query clips. Fastest clean-room
  Python reference.
- **kushn/signals-systems-project**: measured numbers — pair-hashing gives an
  **18× larger separation** from the runner-up than single peaks; correct
  down to −5 dB SNR; <100 ms end-to-end after decode. Uses **fan-out 15**
  (repo uses 8) and `maximum_filter` local-max peaks.
- **thmsgo18/shazam** (modern): CLAP/MuQ embeddings as a FAISS pre-filter with
  spectral fingerprints as the verifier, SQLite pickled-BLOB hash storage,
  FastAPI + React. Pattern: cheap embedding narrows candidates, Shazam hashes
  confirm — best-of-both.

**What the best do that ours doesn't:**
1. `maximum_filter`-style local-maximum constellation (freq×time
   neighborhood) — ours is band-based, which can place a peak at a noise
   shoulder inside an empty band. Add as a peak mode.
2. Fan-out 15 (ours: 8) — bigger offset-histogram separation.
3. `identify()` as a one-call convenience + **batch** matching and directory
   indexing (`build_database.py` pattern).
4. Stored track **duration is actually measured** (ours writes 0.0 — a real
   bug: `add_track` reads 4 s of audio and discards it).
5. Constellation visualization export (the deployed app shows spectrogram +
   constellation + offset histogram — users see WHY a match was made).

**Trash mined:** one blog "Shazam clone" that hashed raw waveform bytes —
not even wrong, ignored.

**Acoustic analysis — best mined:**

- **Essentia** (essentia.upf.edu): `HPCP` chroma → `Key`; `TuningFrequency`
  (exact tuning + cents off 440 Hz); `ChordsDetection`/`ChordsDescriptors`
  (chord sequence from chroma); `RhythmExtractor2013`/`BeatTrackerDegara`;
  `Loudness`; and crucially an **audio-problems** family — clicks, pops,
  discontinuity, gaps, **hum detection**, inter-sample peaks, saturation,
  **SNR estimation**, start/stop cuts, true-peak.
- **librosa**: `beat_track` + `feature.tempo(aggregate=None)` for local tempo;
  `chroma_cqt` on **HPSS-separated harmonic audio** before key detection;
  half/double-time tempo candidates; spectral contrast; tonnetz; MFCC.
- **LUFSNormalizer / podcast_leveler / podtools.cc**: platform targets table —
  Spotify −14 LUFS, Apple Podcasts −16 LUFS, YouTube −14 LUFS, mono podcast
  −19 LUFS, EBU R128 −23, ATSC −24; true peak −1 dBTP; PASS/WARN/FAIL
  compliance checker; two-pass `loudnorm` with `linear=true` (declines gain
  that would breach TP — deviation must be *reported*, not silently accepted).

**What the best do that ours doesn't:**
1. Tuning frequency (cents off A440) — one number DJs and vocalists ask for.
2. Chord detection (chroma-template match over major/minor triads).
3. Audio-quality flags: SNR estimate, hum presence, saturation/clipping,
   start/stop cuts — Essentia's "audio problems" family.
4. Platform compliance table + checker (PASS/WARN/FAIL per platform).
5. HPSS-style harmonic/percussive split before chroma (ours uses raw frames).

---

## 2. dsp.py — native DSP toolbox

**Best mined:**

- **spotify/pedalboard** (6k★, Spotify Audio Intelligence Lab): the effect
  taxonomy to beat — guitar-style (`Chorus`, `Distortion`, `Phaser`,
  `Clipping`), loudness/dynamics (`Compressor`, `Gain`, `Limiter`),
  filters (`HighpassFilter`, `LadderFilter`, `LowpassFilter`), spatial
  (`Convolution`, `Delay`, `Reverb`), pitch (`PitchShift`), lossy
  (`GSMFullRate`, `MP3Compressor`), quality (`Resample`, `Bitcrush`).
  Chains = `Pedalboard([...])` (our `EffectChain` mirrors this exactly).
  GIL-released, VST3/AU plugin loading, `pedalboard.io` format I/O with O(1)
  on-the-fly resampling.
- **Auphonic** (the service, not code): adaptive leveling between speakers
  (dynamic normalizer — atkAudio's atkAutoleveller does this in real time
  with input/output limiters, threshold, hysteresis), noise+reverb+hum
  reduction, multitrack **ducking and crosstalk removal**, per-platform
  loudness targets, batch/watch-folder mode.
- **ai-content-factory audio-post-production skill**: the full pro workflow —
  *analyze before editing* (deep/noise/filler/events/emotion/structure
  reports), *plan the chain* (NL intent → operations), repair (denoise,
  tighten silences, cut ranges, **device simulation for deliberate lo-fi**),
  level (match level, dialogue leveling, **gain automation**, auto-EQ,
  **duck bed under dialogue**, crossfade joins), master for destination,
  verify (quality check, loudness report, **A/B compare**, export package),
  and a **self-correct loop** that re-measures and fixes.

**What the best do that ours doesn't:**
1. Chorus, Distortion (waveshaper), Phaser (LFO allpass), Tremolo,
   Delay-with-feedback, Highpass/Lowpass biquad filters, Bitcrush/lofi
   device simulation, Telephone (bandpass) — half the pedalboard taxonomy
   is missing from our EFFECTS.
2. Adaptive leveling (RMS-windowed dynamic normalizer) — ours only has
   static peak normalize.
3. **Ducking**: bed ducks under voice (Auphonic multitrack pattern) — ours
   only has `mix_under` (static gain).
4. Named **style presets** ("Podcast", "LoFi", "Telephone", "Radio",
   "Studio", "Club") — users pick a vibe, not a knob list.
5. The full analyze → plan → repair → level → master → verify → export
   workflow as one call (the "make this sound finished" button).

---

## 3. edit.py — transcript-as-timeline editing (Descript pattern)

**Best mined:**

- **Descript** (gold standard): text-based editing where deleting a sentence
  cuts the media; **Overdub** (voice clone fills typed corrections);
  **Studio Sound** (one-click denoise + loudness to platform standards);
  one-click filler removal; bulk **silence removal** with a threshold knob;
  edit modes — **Correct Text** (fix transcript without touching media),
  **Ignore/strikethrough** (non-destructive mute), **Restore removed media**
  (source never destroyed); **Patient Playback**; **Word gap control**
  (tune silence between words at edit points); **Wordbar** (draggable word
  boundaries); multitrack; Underlord AI co-editor; translation dubbing.
- **cassiocassio/bristlenose design doc**: comparative matrix — Trint's
  strikethrough-as-first-class, Otter's re-alignment warnings. Key lesson:
  text edits that move words need re-alignment; deletions need timecode
  honesty.

**What the best do that ours doesn't:**
1. **Silence tightening** (bulk gap removal with threshold) — the #2
   Descript win after fillers.
2. **Move edits** (cut a range, paste at another position — "move a
   paragraph, the audio moves with it").
3. **Silence/mute edits** (strikethrough-as-first-class: region replaced
   with silence, non-destructive preview).
4. **Correct-text mode** (fix transcript text only, media untouched).
5. **Dry-run preview** of an edit list (see what would be cut before
   rendering — Descript's whole UX is "read first").
6. More filler languages — Descript transcribes 22+; ours covers 6.
7. NL intent only knows fillers; Descript's Underlord routes *any*
   audio intent ("tighten the silences", "make it louder", "add reverb").

---

## 4. audiobook.py — EPUB → audiobook

**Best mined (rules checked against Sep–Oct 2026 sources):**

- **ACX/Audible (Sep 2026)**: Audible's own **Virtual Voice** beta program —
  ACX terms **reject third-party AI-generated voices** (Jason Lehigh:
  "they will automatically reject these submissions"). Audible marks
  AI-narrated titles itself. *Our current rule ("ACX requires AI-narration
  disclosure") understates reality — for Devon's cloned-voice pipeline ACX
  is effectively BLOCKED, not disclose-and-ship.*
- **Spotify for Authors (Findaway)**: accepts digital narration when you tick
  **"This audiobook uses digital voice narration"**.
- **Kobo Writing Life**: list the narrator as **"Synthesised voice"**.
- **Author's Republic**: does **not allow any AI-narrated components** — book
  can be removed, royalties withheld. *We don't list them at all — a gap.*
- **narrationbox.com** (quality gates): ACX specs — noise floor **< −60 dB**,
  **RMS −18 to −23 dB**, consistent spacing/chapter formatting; the real
  killers are flat delivery, **inconsistent pacing between chapters**, accent
  drift, mispronunciations. Findaway/Spotify/Apple monitor **completion
  rates**, not just specs.
- **Vois tutorial (Sep 2026)**: ACX Audiobook preset = **−20 LUFS, MP3
  192 kbps** export. *Our pipeline masters everything to −16 LUFS — wrong
  for ACX-shaped targets.*

**What the best do that ours doesn't:**
1. Per-store **mastering targets** (ACX RMS −18…−23 vs podcast −16 LUFS),
   not one hardcoded −16.
2. Store **allow/block rules** (Author's Republic = refuse; ACX = virtual-voice
   only → refuse our cloned voice with the honest reason), versioned and
   re-checked.
3. **Compliance checker**: measure noise floor + RMS + true peak → PASS/WARN/
   FAIL per target (podcast_leveler pattern).
4. **Chapter pacing consistency** check (RMS drift across chapters — a known
   rejection trigger).
5. ACX export preset: **MP3 192 kbps** final package.

---

## 5. pipeline.py — unified voice pipeline (WhisperX pattern)

**Best mined:**

- **WhisperX** (m-bain/whisperX): transcribe (faster-whisper) →
  **wav2vec2 forced alignment** (word-level timestamps) → **pyannote
  diarization** (speaker-diarization-3.1 + segmentation-3.0, HF token +
  EULA) → `assign_word_speakers` (timestamp-overlap so labels respect word
  boundaries). Speaker labels survive mid-segment switches.
- **whisperx-transcriber** (0xkaz): the same core as a CLI — outputs
  **SRT, VTT, TXT, JSON, Markdown meeting minutes** (speaker-grouped turns).
  Fully local, no API key on the transcribe→align path.
- **aarondodd/meeting-notetaker** issue #8: ECAPA-TDNN vs pyannote tradeoff
  table — real diarization costs ~35% wall-clock on a 30-min meeting but
  drops DER under 5%; honest about the HF-token friction.
- **jamditis journalism SKILL.md**: stereo-with-bleed is worse for
  diarization than clean mono; `min_speakers`/`max_speakers` knobs matter.

**What the best do that ours doesn't:**
1. Real diarization stage (pyannote/WhisperX) when available — ours is
   honest but permanently single-speaker; it should *try* the real thing
   and degrade honestly.
2. **Meeting-minutes output** (speaker-grouped turns with timestamps).
3. **SRT/VTT export** from the word timings we already have.
4. min/max speaker knobs passed through.

---

## 6. overview.py — interactive audio overviews (NotebookLM pattern)

**Best mined:**

- **NotebookLM Audio Overview + Interactive Mode (Join)**: two AI hosts,
  10–20 min deep dives; **Join button** — interrupt mid-playback, the hosts
  invite you in, answer from the uploaded sources, **adjust their
  explanation to your confusion**, keep conversational continuity. Users
  steer with grounded questions ("give me a concrete example") rather than
  accepting explanations. NotebookLM suggests follow-up questions while you
  listen; multilingual; offline download.
- **NotebookLM limitation**: same two default voices; custom voices need
  export + external tools. *Our two-cloned-voice design already beats this —
  say so.*

**What the best do that ours doesn't:**
1. **Suggested questions** (the host suggests what to ask next — drives the
   interactive loop).
2. **Steering with continuity**: follow-up answers reference the discussion
   so far (ours answers each question cold — history exists but is never
   fed back).
3. **Chapter timestamps measured from real audio**, not word-proportional
   estimates (ours estimates — honest label, but real measurement is
   available: we render each line's audio separately).
4. More formats: NotebookLM has Deep Dive / Debate / Brief + **custom
   prompts**; ours should add at least an Interview format.

---

## 7. characters.py — conversational story characters

**Best mined:**

- **Character.AI / roleplay best practice** (aireiter.com, storychat.blog):
  roleplay memory has three layers — conversation history, retrieval,
  character state. The durable pattern is a **Lorebook**: compact pinned
  facts (relationships, injuries, promises, locations, inventory,
  unresolved conflicts) updated *outside* the model's prose, injected
  selectively. Pinning beats ever-growing transcripts.
- **OCD (open-character-design) spec**: character blueprint — identity,
  personality traits, **behavior directives** (tone, improv style),
  interaction layer, **state dynamics** (mood/health/morale evolving with
  the simulation), meta properties. Prompt composition + memory systems +
  state sync as the three runtime patterns.
- **"Magical roleplay prompt"** (medium, enderdragon): inner alignment —
  "internally recall your complete identity… emotional state,
  relationships, memories, goals as if re-entering your own mind";
  *never format like an AI* (no bullet points, no markdown, no pull
  quotes — dialogue as direct texting); subtle emotional subtext, what's
  left unsaid.
- **ourcodeworld persona guide**: tone rules must be *specific*, knowledge
  needs *clear limits*, the model must know when to say it lacks enough
  information; test consistency by asking similar questions different ways.

**What the best do that ours doesn't:**
1. **Lorebook / pinned facts** (relationships, promises, injuries) — ours
   only has chapter memories; relationships aren't modeled at all.
2. **State dynamics** (mood, evolving stance) — ours is static.
3. Conversation history passed into `talk_to` (ours takes one question,
   no history).
4. Inner-alignment + never-format-like-AI prompt lines.
5. Exportable **character card** (the shareable artifact).

---

## 8. Style / presentation gold

- **podcast_leveler**: PASS/WARN/FAIL badges per platform — compliance as a
  glanceable card, not a paragraph.
- **kushn deployed app**: show the constellation + offset histogram — *show
  the work*, not just the verdict.
- **whisperx-transcriber**: one core, many exports (SRT/VTT/TXT/JSON/MD) —
  every artifact the user could want, from one run.
- **LUFSNormalizer**: 10 built-in presets as quick-select buttons; preset
  manager — presets are the UX, knobs are the fallback.

---

## Implementation plan (what this sweep adds)

| Module | Add | Upgrade | Improve |
|---|---|---|---|
| fingerprint | `identify()`, batch match, directory index, track duration fix, tuning (cents), chord detect, quality flags (SNR/hum/sat), compliance table, constellation export | fan-out 15, local-max peak mode option | richer `describe_audio`, analysis dict |
| dsp | chorus, distortion, phaser, tremolo, delay, highpass, lowpass, biquad, telephone, bitcrush, vibrato, `autolevel` (adaptive), `duck_under` (sidechain duck), CHAIN_PRESETS, `EffectChain.from_preset` | enhance profiles (+podcast, +audiobook) | list_fx catalogue |
| edit | `remove_silences`, Edit "move"/"silence" kinds, `correct_text`, `preview_edits` dry-run, fr/es/de fillers, wider `nl_audio_intent` | splice keeps source intact (restore) | god-tier fx catalogue |
| audiobook | Author's Republic + ACX virtual-voice rules, `allowed` flag, per-store targets, `check_compliance`, chapter pacing check, MP3-192 export | rules version bump + recheck date | store list output |
| pipeline | real diarization attempt (whisperx/pyannote) w/ honest fallback, meeting minutes, SRT/VTT export, speaker knobs | diarize stage | engines output |
| overview | suggested questions, steering history, measured chapter timestamps, "interview" format, export markdown | grounding (keep) | list/voices output |
| characters | relationships, lorebook facts, mood/state, history in talk_to, voice reply, character card export | prompt (inner alignment, no-AI-format) | list output |
