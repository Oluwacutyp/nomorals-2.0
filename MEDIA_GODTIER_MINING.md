# Media God-Tier — Mining Report (Phase 8)

**Rule:** mine everything, best AND trash. Written BEFORE build commits.
**Scope:** `nomorals/media/` — 119 Python files across directed/, imggen/,
videogen/, edit_engine/, motion_studio/, scene_intel/, contentops/, plus
root music modules (out of phase-8 scope except where they touch video).

## 1. Inventory — what exists

### directed/ (tonight's system, ~3,200 lines)
| module | lines | does |
|---|---|---|
| motion_score.py | 682 | open-ended LLM-choreographed motion (timed phases, joint targets, easing); physical guards |
| pose_rig.py | 458 | COCO-18 + 21pt-hand parametric rig; keypoint trajectory authoring; pose-video render |
| validate.py | 289 | structural/physical motion validation, 0–100 plausibility |
| lipsync.py | 366 | Wav2Lip/LatentSync/SadTalker wiring + CPU audio-envelope fallback + dub_video |
| camera.py | 335 | phone_selfie / CCTV / dashcam / cinema emulation (shake, rolling shutter, ISP look) |
| prompt_engine.py | 295 | NL → structured backend-optimized prompts (LTX/Wan/motion/image renderers) |
| ai_edit.py | 258 | inpaint/outpaint/face-swap/bg-replace/object-remove/relight/expression |
| filler.py | 150 | frame bridging, background extension, video gap repair |
| animator.py | 218 | MimicMotion neural + CPU mesh-warp fallback, honest routing |

**Status:** all import cleanly. Honest about neural vs CPU. Weaknesses:
- `motion_score` easing library is thin (linear + basic curves only)
- `pose_rig` hand morphs are parametric only — no finger IK solver
- `camera.py` covers 4 looks; missing: drone, bodycam, webcam, vintage film, anamorphic
- `lipsync` CPU fallback is jaw-only (no viseme shapes)
- `prompt_engine` covers LTX/Wan/motion/image; missing backends: HunyuanVideo, Mochi, CogVideoX, Stable Video Diffusion
- `filler.py` gap repair is optical-flow-blind (simple crossfade-ish)
- `animator.py` warp is single-grid; no layered/depth-aware warp

### imggen/ (~3,600 lines)
Hand-rolled DDPM/DDIM schedulers (numpy + torch shim, tested), UNet,
SD-compat loader, LoRA, training loop, pipeline, studio API, upscale.
`edit.py` (img2img SDEdit, blended-diffusion inpaint, outpaint) requires
torch — fails honestly without it.

**Weaknesses:**
- No CPU/PIL fallback for inpaint/outpaint (torch-only) — phone can't edit
- Mask generation is manual (feather_mask) — no auto-mask from text/point
- No ControlNet-style conditioning (pose/depth/edge) in the native stack
- Upscale is basic — no Real-ESRGAN-class wiring check
- img2img has no strength auto-selection from edit magnitude

### videogen/ (~1,000 lines)
LTX + Wan backends, capability probing, chaining (scene stitching),
pipeline. **Weaknesses:**
- Only 2 neural backends; no HunyuanVideo/Mochi/CogVideoX adapters
- Chaining is cut-based; no morph/transition-aware stitching
- No temporal consistency enforcement between chained clips

### edit_engine/ (~2,400 lines)
Timeline, spec, render (ffmpeg), transitions, effects, keyframes, text,
audio. Style-agnostic (fixed per 2026-10-09 directive). **Weaknesses:**
- Effects library is thin vs ffmpeg's actual filter set
- No auto-caption/subtitle burn-in from transcription
- No beat-sync cutting (audio analysis → cut points)

### motion_studio/ (~2,200 lines)
Ken Burns, montage, grading, typography, visualizer, studio API.
**Overlap found:** kenburns/montage duplicate some directed/filler
frame ops; grading duplicates edit_engine/effects color ops. Consolidation
candidate.

### scene_intel/ (~4 files)
Characters, score, segment, track — scene understanding for editing.
Needs wiring audit vs contentops/pipeline.

### contentops/
Niche content pipelines (anime_edits, reddit_stories, etc.), publish
(tiktok/x/youtube), styles (documentary/minimal/phonk/vlog). Out of
Phase-8 core scope but prompt_engine should know their needs.

## 2. Best-in-class outside (what god-tier means)

- **Runway Gen-4 / Google Flow (Veo):** directed motion via text + camera
  language ("dolly in", "orbit left"); temporal coherence; our edge: we
  author the POSE explicitly (deterministic, editable) — they infer it.
  Close the gap with: richer camera language, motion-beat sync to audio.
- **Magnific / Krea (image):** structure-preserving upscales + creative
  reinterpretation slider. Our imggen needs a creativity slider on upscale.
- **Runway inpainting / Adobe Firefly:** mask + prompt + structure match.
  Our blended diffusion is the right core; needs auto-mask + edge-aware fill.
- **Topaz Video:** frame interpolation (RIFE), stabilization, upscale
  chain. Our filler needs RIFE-class interpolation wiring.
- **DaVinci Resolve (edit):** beat-sync, auto-caption, magic mask.
  edit_engine needs: transcription→captions, audio-beat→cut.

## 3. Concrete build list (floor, not ceiling)

### A. Directed animation consolidation + depth
1. Easing library expansion (cubic-bezier, spring, overshoot, anticipation)
2. Finger IK solver for hand morphs (two-bone per finger, tendon coupling exists)
3. Camera pack expansion: drone, bodycam, webcam, vintage film, anamorphic, gimbal
4. Camera language: parse "dolly in", "orbit", "crane up" from NL into camera programs
5. CPU lipsync: viseme shapes (not just jaw) via phoneme→mouth-shape map
6. Depth-aware warp: layered foreground/background warp in animator
7. Motion↔audio sync: beat-grid alignment for motion_score phases
8. Consolidate: motion_studio kenburns/montage frame ops → use directed/filler; dedupe grading vs edit_engine color

### B. Image editing god-tier
1. CPU/PIL fallback for inpaint (telea-style via OpenCV if present, else patch-match-ish numpy) so phone can edit
2. Auto-mask: text/point → mask via simple segmentation (SAM if available, GrabCut fallback)
3. Outpaint quality: multi-scale fill + content-aware edge blending
4. Creativity slider on upscale
5. Strength auto-select for img2img from edit description magnitude
6. ControlNet-style conditioning hooks in pipeline (pose/depth/canny) — wire when diffusers present, define interface now

### C. Video pipeline + prompt engine
1. Prompt engine: HunyuanVideo, Mochi, CogVideoX, SVD renderers; negative-prompt builder per backend; shot-composition vocabulary (rule of thirds, leading lines, depth layers)
2. videogen: temporal-consistency pass between chained clips (color + structure match on boundary frames)
3. filler: RIFE wiring when available, improved fallback interpolation (motion-compensated blend)
4. LTX/Wan: resolution/duration auto-tune from profile (phone vs workstation)

## 4. Trash-build gold

- Warp animator's gaussian-falloff displacement: keep, add depth layers.
- Hand-rolled DDPM math: keep (tested, dependency-free) — it's the fallback spine.
- Honest backend routing ("warp, not neural"): keep as the pattern for every new module.
- Style-agnostic engines + preset consumers: already the rule; prompt_engine must not bake aesthetics.

## 5. Test plan

- New: easing math, finger IK reachability, camera program parsing, viseme mapping, mask feather edges, outpaint seam invisibility, prompt renderer golden outputs, chain boundary consistency metric.
- Existing suites: test_directed.py, test_imggen.py, test_videogen.py, test_media_edit*.py must stay green.
