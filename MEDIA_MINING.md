# MEDIA_MINING.md — external mining for the `nomorals/media/` sweep

Date: 2026-10-10. Method: for every significant class in `nomorals/media/`
(122 files), find how the BEST implementation outside the repo does it,
mine best AND trash, then merge the gold into the real classes. No stubs,
no deletions, no parallel systems.

Each section answers: **how does the best implementation of X do it?**
plus the user's standing questions: **what features SHOULD this have that
it doesn't?** and **how should this LOOK/FEEL when the user interacts
with it?**

---

## 1. Music composition — `MusicCreator`, `Song`, `SongSpec`, `SongDraft`, `Producer`, `ComposerLLM`, `AfrobeatsLoRA`

### Best outside
- **repsac/produzre** (`produzre/composer/`) — the current gold for
  programmatic composition. Inverts the usual mistake (randomness at bar
  level, static structure): **Song DNA** — hook, answer, verse and bridge
  ideas plus a signature-lick bank chosen by a memorability search
  re-scored over the real chorus/verse chords. **Phrase grammars per
  section** (verse period, climbing pre-chorus, A A′ B A″ chorus,
  contrasting bridge, narrative solo, outro) with **section memory**:
  returning choruses repeat note-for-note, the final chorus lifts.
  A **listener expectation model** (primed on human interval statistics,
  ~3.5k phrases) picks developments by target surprise. **Groove memory**:
  medoid source bars + chord-relative restatement and section recall for
  drums/bass/guitar. Cadential turnarounds; opt-in final-chorus key
  change. Determinism doc + musicality benchmark + audio preview renderer.
- **rfhits/SongComposer** (ACL 2025) — LLM for lyric+melody with
  **symbolic song representations**: better token efficiency, precise,
  human-readable; outperforms GPT-4 on lyric↔melody and text→song.
- **neilpontecorvo/songblueprint** — machine-readable blueprint language:
  probabilities over chords/structure from tonal harmony; style/mood fit.
- **andreisminsk/music-gen** (YuE2) — LLM-first pipeline: chain-of-thought
  modes (full = melody + chord planning, melody-only, off); lyrics fed in
  with `[Verse]/[Chorus]` section markers.

### Verdict on ours
Ours is ahead of most trash builds: real LLM-first lyrics
(model writes the song BEFORE rendering — matches the user's standing
music directive), syllable-aware template engine, real SMF MIDI
arrangement, 22 drum patterns, GM programs. **Missing vs the gold:**
no Song DNA, no phrase/section memory (choruses regenerate instead of
repeating), no memorability search for the hook, no listener-surprise
model, no groove memory across sections. `ComposerLLM` is thin.

### Gold merged in this sweep
- `SongDNA` in `music.py`: hook/answer/bridge ideas + signature lick
  bank + memorability scoring; `MusicCreator.compose()` builds DNA from
  topic + style before any section is written.
- Section memory: first chorus's lead line is stored; returning
  choruses replay it note-for-note; final chorus gets a lift (octave +
  velocity + drum fill) instead of a fresh random draw.
- `to_markdown()` restyled via the new theme engine.

---

## 2. Synthesis / mixing — `synth.py`, `synth_backend.py`

### Best outside
- **klunk386/synth8** — modular signal chains: oscillators, filters,
  VCAs, ADSR/LFO modulators; each chain managed as a **voice**; a
  **Mixer class** combines layered voices/chords; dynamic voice
  activation + cleanup.
- **kennycason/kengine** design doc — the canonical signal flow:
  `OSC1/OSC2/noise → mixer → SVF filter → VCA → effects → master`,
  envelopes/LFOs driving filter cutoff + amplitude, velocity sensitivity,
  voice pooling (allocate/route, no allocation in the hot path).

### Verdict on ours
Ours is a solid offline renderer (dual stdlib/numpy paths,
`mix_tracks`, ADSR, kicks/snares/hats) but has no voice abstraction,
no patch concept, no filter stage, no effects bus — it's render
functions, not an instrument.

### Gold merged in this sweep
- `Voice` + `Patch` + `SynthBus` classes in `synth.py` layered OVER the
  existing render functions (no rewrite): a voice owns osc mix,
  ADSR envelope, velocity; a patch is a named preset (bass/lead/pad/
  pluck/stab) with cutoff/resonance/filter-env; the bus adds a
  one-pole lowpass stage + simple feedback delay send. Existing
  `mix_tracks` stays the offline path; the new classes give the
  per-note expressiveness the gold has.

---

## 3. DJ engine — `dj_engine.py`, `dj.py`, `dj_live.py`, `dj_mixdrop.py`, `taste.py`

### Best outside
- **gnujoow/spotify-mixmaster** — the reference set builder: directional
  Camelot scoring (same / ±1 / relative / energy-boost moves), BPM flow
  with **±6% transitions and half/double-time matching** (87≈174),
  **energy curves** (classic late-peak, linear, wave, flat, or custom
  YAML), genre buckets + opener→closer DJ slots, draft workflow
  (landmark tracks pinned, gaps filled around them), **quality floor:
  refuses to force bad transitions — ends the set short instead**,
  every run records its seed (reproducible), all rules in config.
- **hwgogwedez/dj-harmonic-analyzer** — canonical Camelot map, harmonic
  journeys (linear 8A→9A→10A, key-to-key pathfinding on the wheel,
  energy-building paths, mood-shift 8A→8B, zone-based mixing).
- **Mixed In Key rules** (the originators): same key, same number
  different letter, ±1 number — the T-shape.

### Verdict on ours
Ours already has Camelot scoring, BPM detection (onset autocorr),
Krumhansl key detection, phrase-aligned transitions, energy arc —
better than most GitHub DJ toys. **Missing vs gold:** named energy-curve
presets, half/double-time BPM bridging, key-to-key pathfinding on the
wheel, quality floor (ours will force a bad transition rather than stop),
seeded reproducibility, landmark/draft workflow.

### Gold merged in this sweep
- `energy_curve()` presets: `late_peak` (classic), `wave`, `linear`,
  `flat` — `plan_energy_arc` takes `curve=`.
- `harmonic_path(from_camelot, to_camelot)`: BFS shortest path on the
  Camelot wheel (T-shape moves), so the planner can route a journey
  between distant keys instead of jumping.
- Half/double-time BPM compatibility in `plan_transition`
  (87 vs 174 now beatmatches honestly).
- `quality_floor` in `plan_energy_arc`: transitions below the floor are
  refused; the set ends short with an honest reason instead of padding.
- `seed` parameter → deterministic ordering via seeded RNG.
- ASCII journey map in the arc report (god-tier feel).

---

## 4. Vocals — `vocals.py`, `vocal_lite.py`, `afrobeats.py` (NaijaSungVocals)

### Best outside
- **openmontage** provider table: ElevenLabs (SFX/music), RVC voice
  conversion, DiffSinger (singing synthesis), Wav2Lip/MuseTalk/SadTalker
  for visual sync. Pattern: **one scored provider selector** over many
  backends, local-first with cloud fallback, auditable decision trail.
- Trash pattern to avoid: a dozen half-wired adapters with no selection
  logic and silent fallbacks.

### Verdict on ours
Ours has the right shape (`VocalChain` with DiffSinger/RVC adapters,
`RVCVoiceRegistry`, Naija tone-aware sung vocals) — the adapters exist
but selection is manual and there's no ranked provider choice.

### Gold merged in this sweep
- `VocalChain.pick_backend()` — scored backend selection (availability
  → quality → latency), decision logged in the result dict; honest
  `unavailable` reasons instead of silent fallback.
- `render_acapella_preview()` in `vocal_lite.py`: fast offline
  formant-ish preview so the user hears the melody before the heavy
  model runs.

---

## 5. Image generation — `imggen/` (pipeline, diffusion, unet, lora, train, data, sdcompat, masks, edit, studio, upscale)

### Best outside
- **Hugging Face diffusers** design principles (the gold standard):
  **modularity** — pipelines are thin wrappers; model/scheduler/
  processor independently loadable; **component reuse** across
  pipelines (`img2img = StableDiffusionImg2ImgPipeline(**text2img.components)`);
  **scheduler swapping** (`pipeline.scheduler = EulerDiscreteScheduler.from_config(...)`);
  **adapters** (`load_lora_weights` / `unload_lora_weights`, mergeable or
  dynamically switchable); **inference-only contract** (`torch.no_grad`).
- **ComfyUI workflow sequencing** (working artists' pattern): generate →
  img2img refine (denoise/strength controls keep) → inpaint masked area →
  upscale at full size and inspect. LoRA = add-on selected into a
  compatible workflow; ControlNet guides layout (canny/depth/pose);
  **IP-Adapter + ControlNet combo** is the best-practice stack for
  style + structure control.
- Trash: monolithic `generate()` with 40 kwargs, no component reuse,
  LoRA baked in with no unload path.

### Verdict on ours
Ours has a real native UNet + DDPM/DDIM schedulers + ControlNet
adapters + LoRA + masks + edit + upscale + studio — remarkably
complete. **Missing vs gold:** no component reuse across modes
(t2i/img2img/inpaint each rebuild), no one-call refine chain, no
dynamic adapter stack (load/unload/swap), no scheduler swap helper.

### Gold merged in this sweep
- `NativePipeline.components()` — return the reusable component dict
  (diffusers pattern); `from_components()` classmethod rebuilds any
  mode pipeline from it.
- `NativePipeline.chain()` — one call: t2i → img2img refine →
  optional upscale (ComfyUI artist sequence, codified).
- `swap_scheduler(name)` — DDPM↔DDIM↔Euler-style swap without rebuild.
- Adapter stack: `load_adapter()` / `unload_adapter()` /
  `list_adapters()` with per-adapter weight (dynamic, switchable).

---

## 6. Video editing — `edit_engine/` (timeline, render, effects, transitions, keyframes, audio, text, spec)

### Best outside
- **soaing2024/video-studio** `references/pipeline.md` — the gold for
  ffmpeg-based rendering: **one ffmpeg invocation per project**; per
  input `trim/setpts/fps/scale+pad`; fold = first clip is the
  accumulator, cut→`concat`, transition→`xfade=transition=T:duration=D:offset=O`
  with `O = accumulated_length − D`; look chain (eq grade → fades →
  progress bar → subtitles); **audio**: `atrim, asetpts, volume, afade,
  adelay` then `amix`, `atrim` to video length; filter graph written to
  `build/<name>/filter.txt` and passed via `-filter_complex_script`.
- **MastroMimmo/ffmpeg-skill** (`fftools.py`) — structured **JSON output**
  per command, CRF quality guide (18 archival / 23 default / 28 social),
  smart codec selection by extension, size reporting.
- **jianying headless** — portable build binds plan+timeline+resources
  by SHA-256; result asserts duration/dims/fps/frame count/H.264/
  yuv420p; keeps filter graph + ffprobe evidence.
- Trash: one ffmpeg call per clip then concat (generation loss, slow),
  transitions as hard cuts labeled "crossfade".

### Verdict on ours
Ours has Timeline/Clip/Track/Transition/Keyframes and an ffmpeg
renderer — real. **Missing vs gold:** single-invocation guarantee isn't
explicit, no CRF presets, no structured render report (JSON), no
post-render assertion pass.

### Gold merged in this sweep
- `render_timeline(..., single_pass=True)` made explicit + `CRF_PRESETS`
  (`archival`/`high`/`balanced`/`social`/`draft`).
- `RenderReport` dataclass: JSON-serializable (inputs, filter graph
  path, duration, dims, fps, size, crf, sha256) — fftools-style honesty.
- `self_review()` on the report: ffprobe asserts duration/dims/fps/
  frame count; returns pass/fail per check (jianying pattern).

---

## 7. Video generation — `videogen/` (pipeline, wan_backend, ltx_backend, chaining, consistency, autotune, capabilities)

### Best outside
- Ours already covers: capability report, neural generate with
  LTX/Wan backends, motion fallback, `request_hero_clip`, scene
  chaining, boundary consistency (histogram correlation + edge
  similarity), Reinhard color matching, grade-head pass.
- Gold pattern from chaining systems: **per-scene seed propagation**
  and **reference-frame carryover** so chained scenes don't drift.

### Gold merged in this sweep
- `chain_scenes(..., carry_seed=True, carry_reference=True)`: seed
  derived per scene from a master seed (reproducible), last frame of
  scene N becomes the reference/start frame hint for scene N+1,
  recorded in `ChainReport`.

---

## 8. Scene intelligence — `scene_intel/`, `highlight_model/`

### Best outside
- **priyankit07/highlight-studio** — the working recipe, no GPU needed:
  **logarithmic RMS energy envelope at 10 Hz**; **adaptive 90-second
  rolling median baseline** (adapts to loud derbies vs quiet grounds);
  **excitement rise + sustain filtering** (rise ≥ min_rise_db, sustain
  ≥ min_sustain_s; rejects whistles/thumps); **expand-first windowing**
  (pre_roll lead-in + post_roll aftermath applied BEFORE merging, so
  zero duplicate/overlapping cuts); **greedy budget assembly** ranked by
  acoustic energy area up to a target length, joined with 0.25 s fades.
  Output: reel MP4 + JSON/CSV manifests + plot.
- **artkulak/twitch-stream-highlights** — multi-feature metamodel
  (audio + motion + chat), logistic regression for interpretability.
- **ClaudeCodeCafe/vshot** — ffmpeg scene detection for keyframes,
  montage grids for token-efficient review.

### Verdict on ours
Ours is genuinely multi-modal (audio 40 / motion 30 / faces 20 /
dialogue 10, learned MLP override, adaptive peaks) — ahead of the
audio-only trash. **Missing vs gold:** expand-first windowing,
greedy budget assembly into an actual reel plan, per-event
rise/sustain parameters, manifest output.

### Gold merged in this sweep
- `excitement_events()` in `scene_intel/score.py`: log-dB envelope,
  rolling-median baseline, rise+sustain gating, expand-first
  pre/post-roll windows, overlap merge.
- `assemble_highlight_reel()`: greedy ranking by energy area up to a
  target duration, fade plan, JSON manifest — the highlight-studio
  pipeline in our multi-modal scoring.

---

## 9. Motion studio — `motion_studio/` (core, grading, kenburns, montage, studio, typography, visualizer)

### Best outside
- The user's bar (standing): never simplify to fix a bug; UI controls
  must work for real. Best kinetic-typography tools: word-level timing,
  phrase grouping, easing per word, background treatments (solid/
  gradient/blur), safe-area awareness.
- Trash: subtitle burn-in labeled "kinetic typography".

### Verdict on ours
Ours has word-level `Word` timing, presets, lyric renderer with
backgrounds, quote cards, Ken Burns, visualizer, grading — real.

### Gold merged in this sweep
- Typography: per-word easing styles + emphasis keywords
  (auto-highlight of hook words), safe-area presets for
  9:16 / 16:9 / 1:1.
- `studio.py`: `quick_montage()` one-call path (clips + beat grid →
  cut on beats with xfade) — the missing "make me a montage" verb.

---

## 10. Content ops — `contentops/` (pipeline, audio, beats, edit, niches, publish, styles)

### Best outside
- **victorhugo/openmontage** — agentic production done right: 100+
  tools behind **one scored provider selector** (7 dimensions: task
  fit, quality, control, reliability, cost, latency, continuity);
  **production knowledge files** (directors, checklists, quality
  gates); **pre-compose validation** before GPU spend; **mandatory
  post-render self-review** (ffprobe + frame extraction + audio
  analysis) so garbage is never presented; **auditable decision
  trail**; budget governance (cost estimate before execution).
- **sagearbor/taskcaster** — injectable analyzers; decision order
  unit-testable without ffmpeg.
- Trash: linear script with no validation, silent fallbacks, no
  manifest.

### Verdict on ours
Ours has the pipeline skeleton (Job/JobStore, ShortPipeline, niches,
publish ledger, styles) — the structure exists. **Missing vs gold:**
quality gates, pre-flight validation, post-render self-review,
decision trail.

### Gold merged in this sweep
- `ShortPipeline` gains `preflight()` (asset checks before render
  spend) and `self_review()` (ffprobe-based assertions on the output)
  with results recorded on `RunResult` — the openmontage gate pattern.
- `RunResult.to_dict()` now carries a `decision_trail` list.

---

## 11. Directed video — `directed/` (pose_rig, lipsync, ai_edit, camera, filler, animator, motion_score, prompt_engine, validate)

### Best outside
- openmontage analysis tools: Wav2Lip / MuseTalk / SadTalker for
  lipsync; WhisperX word-level timestamps; CLIP/BLIP-2 video
  understanding; Perlin-based handheld camera feel.
- Ours has Perlin1D camera shake presets, lipsync adapters,
  motion scoring, prompt engine, validation — the pieces exist.

### Gold merged in this sweep
- `camera.py`: `CameraProgram.to_ffmpeg()` — export the program as a
  real ffmpeg `zoompan`/crop chain so the program actually renders
  (closes the plan→render gap).
- `validate.py`: `validate_plan()` gains severity levels + fix
  suggestions (not just pass/fail).

---

## 12. Playback / library / distribution — `playback.py`, `library.py`, `downloader.py`, `resolver.py`, `sources.py`, `film_sources.py`, `distribute.py`, `picklist.py`, `cookies.py`, `caps.py`, `MediaHub`

### Best outside
- fftools: JSON output, smart codec selection, size reporting.
- spotify-mixmaster: draft/landmark workflow, quality floor honesty.

### Verdict on ours
Solid: mpv IPC playback, library with metadata, resolver with
multi-source fallbacks, write caps, MediaHub façade.

### Gold merged in this sweep
- `MediaHub` exposes the new verbs: `highlight_reel()`,
  `montage()`, `harmonic_journey()`, `refine_image()`.
- `distribute.py`: `ReleasePacket` gains platform spec presets
  (loudness targets per platform: YouTube −14 LUFS, Spotify −14,
  TikTok −14, Club −8) — the missing "master for platform" step.

---

## 13. Style / presentation (new directive)

**What SHOULD the output look like?** Today most reports are flat
dicts or plain markdown. God-tier = scannable at a glance in chat:
section banners, key-value cards, status lines with honest state
words, one ASCII visualization per complex object (Camelot journey,
energy curve, timeline), themeable (electric ninja default, plain,
minimal). **New module `nomorals/media/style.py`**: `Theme`,
`banner()`, `card()`, `kv()`, `bar()`, `journey_map()` — used by
`Song.to_markdown`, DJ arc reports, render reports, highlight
manifests. No parallel system: it's the single formatting spine every
media report funnels through.

---

## Trash patterns deliberately NOT merged
- Monolithic 40-kwarg `generate()` (imggen) — kept our structured
  `PipelineConfig` + chain API instead.
- Silent fallbacks anywhere — every new path degrades honestly with
  a named reason (house rule).
- Hardcoded regex/keyword routing — none added; selection is scored
  or deterministic theory (Camelot), never vibes.
- "For later" parking — highlight learned-weights path, provider
  selection, and adapter stacks ship now.
