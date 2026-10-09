# Movie Scene Intelligence + Generative Video — Mining Report

**Date:** 2026-10-09
**Rule:** mine everything, best AND trash. Gold is gold. No "for later."
**User correction (standing):** "If a model can be used why is it 'for later'? That's holding back." Models get built/wired NOW.

---

## 1. Scene Boundary Detection

### PySceneDetect (the standard)
- **What:** `ContentDetector` (HSV histogram diffs), `ThresholdDetector` (fade to/from black), `AdaptiveDetector` (rolling average, handles gradual transitions).
- **Gold:** AdaptiveDetector is the right default — handles dissolves/fades that ContentDetector misses. The arxiv paper (2506.00667) confirms adaptive threshold strategies beat fixed thresholds across domains without tuning.
- **Take:** PySceneDetect AdaptiveDetector as the segmentation backbone. Pure algorithmic, no model needed, CPU-friendly.
- **Leave:** Nothing. It's the right tool.

### Cadrivyn Scene (trash-with-gold)
- **What:** One-person tool for screenshot extraction. PySceneDetect + coverage-first principle (split long scenes into 5s review intervals).
- **Gold:** The coverage-first principle — a 20s "scene" can contain multiple cool moments. Don't treat detected scenes as atomic; sub-segment for highlight mining.
- **Take:** Sub-scene windowing for highlight extraction.

---

## 2. Person Detection + Tracking

### YOLOv8/v11 (detection)
- **What:** Ultralytics YOLO. `yolov8n` is ~6MB, runs on CPU. Person class is well-trained.
- **Gold:** yolov8n for phone/CPU, yolov8m/l for workstation. Proven in every tracking pipeline surveyed.
- **Take:** YOLOv8n as default person detector. Model: `yolov8n.pt` (~6MB, auto-downloaded by ultralytics).

### ByteTrack / DeepSORT (tracking)
- **What:** ByteTrack associates detections across frames using motion + IoU. DeepSORT adds appearance features.
- **Gold:** ByteTrack is simpler and more robust for our use (offline film analysis, not real-time). The player_reid project uses ByteTrack + OSNet successfully.
- **Take:** ByteTrack-style association. For film (offline), we can run detection at 2-4 fps and interpolate — massive speedup.

### OSNet (person re-identification) ⭐ THE MODEL
- **What:** Omni-Scale Network, **2.2M parameters** — tiny. Pretrained on MSMT17 via torchreid. Extracts appearance embeddings; cosine similarity re-identifies across scenes.
- **Gold:** This is the character-tracking model. 2.2M params means it runs on the phone. Every surveyed pipeline (rap-cv-task, player_reid, person_reid_tracking) uses OSNet + YOLO + ByteTrack.
- **Take:** Wire OSNet NOW via torchreid (`osnet_x1_0` pretrained). Character = cluster of re-ID embeddings across the film.
- **Model:** `osnet_x1_0_msmt17` (~9MB). License: check torchreid (MIT-ish, verify).

---

## 3. Highlight / Cool-Scene Scoring

### Multi-modal scoring (the pattern across ALL surveyed builds)
Every working highlight system combines:
- **Audio energy** (40%): RMS energy, excitement peaks, adaptive rolling baseline (highlight-studio's approach: 90s rolling median, rise-above-baseline + sustain filtering)
- **Motion intensity** (30%): frame-diff / optical-flow magnitude peaks (gaming-moment-detector)
- **Face/person presence** (20%): more faces + longer screen time = more important
- **Dialogue density** (10%): Whisper transcription → words-per-minute spikes = quotable moments

**Gold:** The weighted multi-modal score. No single signal is enough. The football-highlight framework proved audio+visual+context beats any single modality.

### QVHIGHLIGHTS (academic, heavy)
- **What:** Transformer model for moment retrieval + highlight detection via natural language queries. Trained on QVHighlights dataset.
- **Gold:** The *idea* of query-based highlight retrieval ("find the fight scenes"). The dataset could train our small model.
- **Leave:** The full transformer — too heavy for phone. Take the concept, build lighter.

### The small trainable model (BUILD NOW, per user)
- **Task:** Given per-second features (audio_energy, motion, face_count, dialogue_density, shot_change_rate), output highlight_score 0-1.
- **Architecture:** Tiny MLP or 1D-CNN, <100K params. Trainable on QVHighlights (public) or on owner feedback (their "cool" picks = labels).
- **Training path:** QVHighlights saliency annotations → feature extraction → train MLP → export ONNX. Owner's picks fine-tune it.
- **This is NOT "for later."** Build the feature pipeline + model scaffold + training script now. Ship with heuristic weights; the model slots in when trained.

---

## 4. Film Download Sources (Nigerian context)

| Source | Content | Structure | Notes |
|---|---|---|---|
| NetNaija (thenetnaija.net) | Nollywood + Hollywood + music | WordPress `?s=` search, post pages | Already have music source; extend to /movies/ |
| Nkiri.com | Nollywood/Hollywood/Korean | Direct download links | Free, no registration |
| FzMovies | Hollywood/Bollywood | Mobile-friendly, direct links | Well-known, stable |
| Toxicwap | Movies, series, Korean | Direct download | No registration |
| 36vibes, SeriezLoaded | Mixed | Blog-style | Fallbacks |

**Gold:** These are all scrape-friendly (no API, no auth). The existing `BaseScrapeSource` pattern in sources.py extends cleanly.
**Take:** Build `FilmSource` classes for NetNaija-movies, Nkiri, FzMovies. Reuse the downloader. Mark as UNVERIFIED until live-tested (site structures change).

---

## 5. Generative Video (existing + gaps)

### What exists
- **LTX backend:** t2v, i2v, extend modes. Real diffusers pipeline.
- **Wan backend:** alternative neural backend.
- **Motion studio:** CPU fallback (kenburns, montage — not generative).
- **Chaining:** multi-scene generation with prompt suffixes.

### Gaps to build NOW
- **vid-to-vid:** No true v2v. Build it: frame extraction → img2img on keyframes (or all frames at low fps) → temporal smoothing → reassemble. LTX img2img pipeline can do per-frame; interpolate between keyframes for speed.
- **As editing ops:** t2v/i2v/v2v must be timeline operations — `generate_shot(prompt, duration)` returns a clip that drops into the edit timeline, gets graded/cut/transitions like any footage. Not a separate toy.

---

## 6. Trash Builds (discipline lessons)

- **rap-cv-task:** Proves the full YOLO+OSNet+DeepSORT pipeline works on commodity hardware. Copy the architecture, not the code.
- **gaming-moment-detector:** Motion-intensity peak detection is embarrassingly effective for action. Don't over-model what DSP finds.
- **video-findings:** "Motion peaks locate candidates but don't identify semantic events." The model scores; the human (or LLM) names. Keep the LLM in the loop for *what* the moment is, not *where*.

---

## Build Plan

1. `nomorals/media/scene_intel/` — segmentation (PySceneDetect) + tracking (YOLO+OSNet+ByteTrack) + scoring (multi-modal) + character timelines
2. `nomorals/media/film_sources.py` — NetNaija-movies, Nkiri, FzMovies scrapers
3. `nomorals/media/genedit.py` — t2v/i2v/v2v as timeline editing operations
4. `nomorals/media/highlight_model/` — small trainable highlight scorer + training script (QVHighlights path)
5. Spine tools for all of it
6. `MODELS_AND_PACKAGES.md` updated with new models

**Models wired NOW (not later):** YOLOv8n (person), OSNet (re-ID), PySceneDetect (algorithmic). Highlight MLP scaffold + training path defined.
