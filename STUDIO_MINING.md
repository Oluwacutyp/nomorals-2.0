# Devon Studio — Mining Report

**Date:** 2026-10-09
**Rule:** mine everything, best AND trash. Gold is gold.
**Purpose:** ground the Devon Studio rebuild in real implementations before writing code.

---

## 1. Olive (open-source NLE, C++/Qt6, GPLv3)

**What it is:** Node-based non-linear editor. 0.2 uses a directed acyclic graph (DAG) for compositing, audio, and video processing instead of fixed effect stacks.

**Gold:**
- **Node DAG compositing** — effects as connectable nodes, not a fixed pipeline. Non-destructive, branchable, mergeable. Nodes copy-paste as *text* — shareable, versionable.
- **OpenColorIO end-to-end color management** — match multi-camera footage with minimal effort; export to any color space (broadcast or web).
- **In-memory timeline structures** — `Sequence` / `Track` / `ClipBlock` / `GapBlock` / `TransitionBlock` / `Footage` / `TimelineMarker`. Clean separation of editorial structure from media.

**Take:** The node-DAG *idea* for complex effect chains (branch/merge/parameterize). The OCIO lesson: color management as a pipeline stage, not an afterthought. Text-serializable effect graphs.

**Leave:** The C++/Qt desktop app itself. The alpha instability. GPLv3 copyleft (we drive tools at arm's length instead).

---

## 2. MLT Framework + Shotcut (the engine behind Shotcut/Kdenlive)

**What it is:** MLT is a C framework (LGPL) with a producer → filter → consumer pipeline. `melt` is its CLI. Shotcut (11k stars, GPLv3) is the Qt6 UI on top.

**Gold:**
- **Producer-filter-consumer model** — producers load media, filters transform, transitions blend, consumers render. Every complex edit is a composition of these four primitives.
- **`melt` CLI** — scripted editing: `melt clip1.mp4 -mix 25 -mixer luma clip2.mp4 -consumer avformat:out.mp4`. The entire timeline is expressible as a command.
- **~150 filters, each in its own folder** — Shotcut's `src/qml/filters/` pattern: one directory per filter with its own UI definition. Scales without a god-file.
- **Undo/redo as command objects** — `src/commands/` (TimelineCommands, FilterCommands, PlaylistCommands). Every mutation is a reversible command.
- **Keyframe models** — per-filter keyframe tracks with interpolation types.
- **Background job system** — encoding/transcoding as jobs, UI never blocks.
- **frei0r plugin ecosystem** — open video-effect plugin standard MLT consumes.

**Take:** The four-primitive pipeline model. The melt-CLI-as-timeline-serialization idea (our ffmpeg filter chains are the equivalent). One-folder-per-filter organization. Command-pattern undo.

**Leave:** Linking libmlt (license drag). The Qt UI. We drive ffmpeg/melt as separate processes.

---

## 3. vean (agent-native video core, TypeScript, AGPL-3.0)

**What it is:** A typed document + edit algebra + diagnostics layer on top of MLT — described as "a language server for video."

**Gold:**
- **Typed document** — the timeline is a typed, serializable document. Parse/serialize/keyframes/edit-algebra/diagnostics all operate on it.
- **Human gestures and agent actions are the SAME operations** — both update the same document, both get undo. An agent calling `apply-op` and a human dragging a clip go through one path.
- **Arm's-length engine driving** — never links GPL code; drives `melt` as a separate process via the public `.mlt` XML format and CLI args. License-clean by architecture.
- **Action runtime** — one typed registry projected to CLI, MCP tools, and LSP code actions. Every public action reachable every way.

**Take:** This is the single most relevant architecture for Devon. The spine (tool loop) calling typed edit operations on a document, rendered by ffmpeg at arm's length. The "language server for video" diagnostics idea — ambient feedback on the edit.

**Leave:** The TypeScript stack. The Mac-app focus.

---

## 4. Kdenlive (KDE NLE, MLT-based)

**What it is:** The most feature-complete open-source NLE. 232 video effects, 242 audio effects.

**Gold:**
- **Proxy editing** — auto-generate low-res proxies, edit smoothly on weak hardware, render full-res. *This is the phone story.*
- **Keyframeable everything** — linear, discrete, and smooth-curve keyframe types per effect parameter.
- **Scopes** — histogram, waveform, vectorscope, RGB parade. "Match scopes, not eyes."
- **Auto subtitles** — VOSK/Whisper integration, SRT/ASS output.
- **Non-blocking render** — separate process, pausable/restartable.
- **Timeline preview render** — pre-render heavy sections for smooth playback.
- **dcc-mcp-kdenlive** — an MCP adapter letting AI agents author Kdenlive projects via typed tools. Dynamic catalog: exposes *installed* MLT services with native parameter metadata, never assumes plugins exist. Every document edit creates a new file (never mutates in place).

**Take:** Proxy editing (mandatory for Termux). The scope set. The dynamic-catalog pattern — expose what the host *actually has*. The MCP-agent adapter pattern validates our spine-tools approach.

**Leave:** The KDE/Qt dependency weight. 474 effects is breadth without curation — we want fewer, deeper.

---

## 5. CapCut (ByteDance — the automation benchmark)

**What it is:** The mobile-first editor that defined auto-editing UX. Closed source; architecture inferred from behavior and docs.

**Gold:**
- **Auto captions pipeline** — multistage: audio analysis → phoneme detection → pretrained transcription (Whisper-class) → waveform alignment → pacing-aware line splitting (2–4s lines) → template engine mapping captions to anchor points (never overlapping key visuals) → legibility check via edge detection + color sampling.
- **Beat sync** — BPM scan → map cuts/zooms/transitions to downbeats. "Smart sync": recalculates the whole timing map when the audio changes. Every edit treated as a unique performance, not a static template.
- **Auto reframe** — subject/movement tracking, keeps key elements centered across aspect ratios, with stabilization.
- **AutoCut** — silence + pause detection, key-scene detection, filler trimming.
- **Template engine** — font hierarchy, animation triggers (text fades on sound cues), color schemes sampled from the video's dominant hues. Trend-styled caption presets.
- **Realtime adaptation** — extend a clip 2s and the system rebalances transitions/zooms to preserve rhythm. A feedback loop, not a one-shot.

**Take:** The caption pipeline stages (we have STT + text layers; the anchor-point and legibility stages are the gap). Beat-synced cut mapping (we have BPM detection in dj_engine — reuse it). The *recalculation* idea: automation that re-derives the edit when inputs change. Auto reframe via subject tracking.

**Leave:** The cloud dependency. The ByteDance data flywheel. Closed source.

---

## 6. FFmpeg filter arsenal (the render substrate)

**What it is:** ~500 filters. The actual pixels.

**Gold (color):**
- `colorbalance` — shadows/mids/highlights per channel (rs/gs/bs, rm/gm/bm, rh/gh/bh), −1..1. This IS lift/gamma/gain.
- `curves` — m/r/g/b point curves. DaVinci-grade precision.
- `lut3d` / `haldclut` — 3D LUT application. `.cube` in, graded pixels out.
- `eq` — brightness/contrast/saturation/gamma in one.
- `colorlevels`, `colorchannelmixer`, `colormatrix` — the rest of the primary toolkit.
- `ciescope`, `datascope`, `vectorscope` — scopes as filters.

**Gold (motion/stabilize):**
- `deshake` — single-pass stabilization.
- `vidstabdetect` + `vidstabtransform` — two-pass, the pro path (needs `--enable-libvidstab`).
- `minterpolate` — motion-interpolated slow motion.

**Gold (edit):**
- `silencedetect` / `silenceremove` — the automation primitives.
- `select='gt(scene,0.4)'` — scene-cut detection as a filter.
- `xfade` — 40+ transitions.
- `drawtext` — burned-in captions with full styling.
- `cropdetect`, `blackdetect` — analysis filters.

**Take:** Everything. This is the render engine; our job is the *interface* (typed ops → filter graphs), not reimplementation. The yomaser/ffmpeg-skill pattern is instructive: typed flags folded into one fixed filter chain, never raw filter strings from callers.

**Leave:** Nothing. But respect that some filters need compile flags (vidstab) — detect and degrade honestly.

---

## 7. KomfyEdit (open-source web editor)

**Gold:**
- **Unified color pipeline** — the SAME `.cube` table drives the WebGL preview shader AND the ffmpeg `lut3d` export. What you see is what renders. Preview/export parity is the feature.
- **Adjustment layers** — one grade applied across all clips below it.
- **Inspector parameter model** — exposure/brightness/contrast/highlights/shadows/temperature/tint/saturation, each with sane ranges.

**Take:** Preview/export parity as a hard requirement. Adjustment-layer concept for timeline-wide grades.

**Leave:** The web stack.

---

## 8. soaing2024/video-studio (measured pipeline)

**Gold:**
- **Measured, not guessed** — vignette/grain numbers measured on a flat grey field. Presets stop where measurement says they should.
- **Finish chain order** — `lut3d → tone (lift/roll/gamma/saturation) → halation → chroma → vignette → grain → unsharp`. Fixed order, every key overridable.
- **"After the fold"** — the look is applied AFTER the montage is assembled, so cuts don't change the look.
- **Shutter** — motion blur by rendering at fps×N and folding with tmix. A delivery decision, cost-linear.

**Take:** The finish-chain order as our default grade pipeline. The "after the fold" rule. The measure-don't-guess discipline for presets.

**Leave:** Nothing — this is all directly applicable.

---

## 9. DaVinci Resolve (the color benchmark)

**What it is:** Industry-standard color. Node-based (serial nodes, each one purpose).

**Gold:**
- **Node order** — white balance → exposure → conversion LUT → wheels (lift/gamma/gain) → saturation → NR → blacks → vignette → sharpen (ALWAYS last). Skip unneeded nodes.
- **Qualifiers (HSL secondary)** — isolate a color range (skin tones) and correct only it.
- **Power windows** — localized shapes for targeted exposure/color.
- **Style-match from reference** — vision model reads a reference frame (color cast, contrast curve, saturation, skin treatment, grain) → translates to concrete grade values → applies. *This is the spine + vision + color tools working together — exactly our architecture.*
- **"Match scopes, not eyes"** — waveform/vectorscope-driven decisions.
- **DRX stills** — grades saved as portable stills, applicable to any clip.

**Take:** The node order as our grade pipeline default. The reference-match workflow (vision → values → apply) as a spine-driven feature. Qualifier/power-window as *concepts* (ffmpeg `geq`/masked filters can approximate).

**Leave:** The desktop app. The Studio-only features.

---

## 10. Auto-editing systems (opencut, auto-editor, ClipsAI, video_editing_agent)

**Gold:**
- **opencut** (Premiere extension, local-first):
  - LLM highlight extraction with *engagement scoring* — hook strength, emotional intensity, pacing, quotability. Multi-dimensional, not one score.
  - Emotion-based highlights via facial analysis (deepface + OpenCV).
  - Shorts pipeline: transcribe → highlight → trim → face-reframe → caption burn-in → export. One click, multi-stage.
  - AI B-roll planning: transcript analysis finds insertion points (dialogue gaps, topic shifts, visual references).
  - Multicam auto-switch via speaker diarization.
  - **OTIO export** — edits as OpenTimelineIO, portable to Resolve/FCP/Avid.
- **auto-editor** — the silence/dead-space removal reference implementation.
- **PySceneDetect** — the shot-detection reference.
- **video_editing_agent** (CrewAI + FFmpeg) — multi-agent pipeline with *complexity routing*: 2 agents for simple (trim/silence/subtitles), 3 for standard, 6 for full (orchestrator, audio intel, scene detect, trimmer, narrative structurer, platform adapter). Natural-language editing.

**Take:** The engagement-scoring dimensions for highlight extraction. The shorts pipeline stage order. OTIO as an interchange format (we should emit it). Complexity-routed agent pipelines — simple tasks shouldn't spin up the full machine.

**Leave:** The CrewAI framework. Premiere lock-in.

---

## 11. Blender VSE (the sleeper)

**Gold:**
- **Strips + modifiers** — modifiers execute *before* strip transform. Mask inputs from other strips.
- **Proxy system** — EXR/float proxies with DWAA compression, no precision loss.
- **Opaque strip detection** — skip rendering covered content. Free performance.
- **Prefetching** — loop-aware, preview-range controlled.
- **GPU scopes** — waveform/vectorscope fully GPU, histogram 2–6x faster.
- **In-timeline compositing** — the VSE is now a real finisher (Blender Studio mastered a 4K HDR film in it).

**Take:** Opaque-strip skipping (don't render what's covered). The modifier-before-transform order. Proxy discipline.

**Leave:** The Blender dependency itself.

---

## 12. "Trash" builds (deliberately mined)

- **ffmpeg-skill CHANGELOG** — a one-person skill repo that documents *every* gap it closes with the exact filter and option verified against `ffmpeg -h`. The discipline of verifying against the binary, not docs, is the gold.
- **Random GitHub NLE experiments** — the consistent lesson from dead projects: they die on *render* (no engine), not on UI. A typed document with no renderer is a toy. Engine first, always.
- **CapCut clones** — they all copy the buttons but miss the *recalculation engine*. Static templates without realtime adaptation feel dead. The gold is the feedback loop, not the template list.

---

## Synthesis — what Devon Studio takes

| Layer | Take from |
|---|---|
| Edit document | vean (typed doc + edit algebra), Olive (ClipBlock/GapBlock/TransitionBlock) |
| Render | FFmpeg filter graphs, arm's-length (vean licensing pattern) |
| Effect organization | Shotcut (one folder per filter), Kdenlive (dynamic catalog) |
| Undo | Shotcut (command objects) |
| Keyframes | Kdenlive (linear/discrete/smooth), Olive (well-implemented) |
| Color pipeline | DaVinci (node order), KomfyEdit (preview/export parity), video-studio (measured finish chain, after-the-fold) |
| Scopes | Kdenlive/Blender (histogram/waveform/vectorscope/parade) |
| Proxies | Kdenlive/Blender (phone story) |
| Automation | CapCut (caption pipeline stages, beat sync, smart sync recalc), opencut (engagement scoring, shorts pipeline), auto-editor (silence) |
| Agent interface | vean (same ops for human and agent), dcc-mcp-kdenlive (dynamic catalog), video_editing_agent (complexity routing) |
| Interchange | OTIO (opencut), FCPXML (Olive survey) |
| Reference matching | DaVinci style-match (vision → values → apply) |

## On the dedicated model question

**Assessment: no dedicated neural model needed for the studio.**

- **Smart reframing** — face detection (Haar in vision/) + rule-of-thirds energy maps. Algorithmic.
- **Scene detection** — ffmpeg `select=gt(scene)` + PySceneDetect-style histogram diff. Algorithmic.
- **Silence cutting** — `silencedetect`. Algorithmic.
- **Color matching** — histogram transfer / gray-world. Algorithmic.
- **Edit decisions** (what to cut, highlight scoring) — LLM-driven via the spine. The brain already exists; a separate model adds nothing.
- **Caption transcription** — STT already exists (faster-whisper chain).

What *would* justify a model later: neural subject segmentation (vs. face-box reframing), learned highlight scoring trained on the owner's retention data. Neither is the God-tier blocker today. The spine + algorithms cover it.

---

## Build order (after this report)

1. Automation (silence, scenes, reframe, batch) → `studio_automation.py` ✓ drafted
2. Keyframes → `edit_engine/keyframes.py` ✓ drafted
3. Pro color (LUTs, curves, wheels, auto-grade, shot match) → `studio_color.py` ✓ drafted
4. Spine tools (`studio_*`) — brain reaches everything
5. Consolidation — deduplicate overlapping stacks, never delete capability
6. Tests per section, commit section-by-section
