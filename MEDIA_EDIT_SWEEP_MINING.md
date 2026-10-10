# MEDIA_EDIT Sweep — External Mining Report

Module: `nomorals/media_edit/` (21 Python files). Mined 2026-10-10 via web search
before any code was written. For each significant class: what the best outside
implementation does, and what our module should take.

## 1. Talking-head avatars (`avatar.py` — LatentSync path)

**Best outside:** ByteDance LatentSync remains the default open-source lip-sync
answer (Apache 2.0, audio-conditioned latent diffusion, SyncNet supervision,
TREPA temporal consistency; ~6.5–18 GB VRAM). Wav2Lip (IIIT-H) is the strongest
lightweight baseline — fast (~25 fps on Apple Silicon), accurate lip-sync but
static head motion; license bars commercial use. SadTalker adds 3D head motion
but quality is inconsistent. EchoMimicV2 adds half-body gestures.
Sources: sync.so LatentSync explainer; flowjam.com AI lip-sync guide;
github.com/metacraft-labs/GuiAssert-Wav2Lip.

**Take:** our module hard-codes one backend (LatentSync). The gold = a backend
registry where Wav2Lip is a real second option (same `talking_head()` surface),
because it runs on CPU-class hardware where LatentSync can't.

## 2. Generative backends (`generate.py`, `comfy.py`, `models.py`, `video_models.py`)

**Best outside:** ComfyUI API format is the lingua franca: CheckpointLoaderSimple
→ CLIPTextEncode → KSampler → VAEDecode → SaveImage; img2img swaps
EmptyLatentImage for LoadImage+VAEEncode; inpainting adds mask input; upload via
`/upload/image`. FLUX + Wan2.2 dominate local txt2img/img2video mindshare;
comfyui-workflow-skill ships 34 templates across SD1.5/SDXL/SD3/FLUX/Wan2.2/
HunyuanVideo/LTXV. Video: Wan 2.2 = best open motion realism (Apache 2.0,
14B/5B/1.3B variants); LTX-2.3 = fastest + only open model with native
synchronized audio (community license, $10M commercial threshold); HunyuanVideo
= cinematic but VRAM-hungry. Instruction editing: MagicBrush-tuned
InstructPix2Pix-class models beat base IP2P on every MagicBrush metric; the
frontier is plan-then-edit (hints/CoT planning + SDXL inpaint).
Sources: vavo/comfyui-codex workflow-tutorials.md; freesosaifared/comfyui-workflow-skill;
crepal.ai LTX 2.3 vs Wan 2.2; apatero.com open video models 2026; arXiv
UltraEdit / PixWizard / MGIE comparisons.

**Take:** we have wan22 t2v workflow but no i2v; video router lacks an LTX route;
prompt handling has no quality-tag enhancement or negative-prompt defaults.

## 3. Face swap (`faceswap.py` — insightface inswapper_128)

**Best outside:** Every serious open face-swap app is insightface `buffalo_l`
detection + `inswapper_128.onnx` + GFPGAN/CodeFormer restoration, then
Real-ESRGAN upscale. The differentiators: multi-face targets (swap every
detected face), glasses/occlusion preservation, embedding averaging for
profile views, colour matching between source/target lighting.
Sources: github.com/yqing999-cmyk/ai-face-swap-web;
github.com/vinothvikas1987/face-swap; github.com/adityaraj8969/headswap;
github.com/saago/faceswap.

**Take:** we swap only the biggest face. Gold = `swap_all` mode, per-face
selection, and a colour-match post step.

## 4. Background removal (`segment.py` — rembg)

**Best outside:** BiRefNet is the open SOTA for edge/hair quality, far ahead of
RMBG-1.4; RMBG-2.0 improves on 1.4 with arbitrary sizes (gated, non-commercial).
`cleanbg` (MIT wrapper over rembg) shows the right API shape: model-per-task
table (`u2net` default, `u2net_human_seg` people, `isnet-general-use` products,
`birefnet-portrait` hair, `isnet-anime` illustration, `bria-rmbg` max quality),
EXIF rotation before inference, `crop`/`background` options on the same call.
Sources: ZhengPeng7/BiRefNet; huggingface.co/briaai/RMBG-2.0;
github.com/fhuayta/cleanbg.

**Take:** we expose one default model. Gold = a model-per-task table with
BiRefNet/isnet options routed through rembg's session mechanism.

## 5. Upscaling (`upscale.py` — Real-ESRGAN)

**Best outside:** Real-ESRGAN family (RRDB) is the default: fast, deterministic,
geometry-preserving. SUPIR (SDXL diffusion) wins on badly degraded photos but is
10–50× slower, needs 8 GB+ VRAM, and *hallucinates text* ("action" → "nstion")
— never point it at documents/diagrams. `spandrel` auto-detects 25+ SR
architectures from a checkpoint (highest-ROI infra decision per
prompt-to-asset research). Asset-type routing: logos/flat-art → edge-preserving
(DAT2/NMKD-Siax/UltraSharp), photos → x4plus/HAT, anime → AnimeSharp.
Sources: tech-insider.org 4K upscaling guide; botmonster.com ESRGAN vs Topaz vs
SUPIR; ai-guru/ai_services compare_upscalers.py; mohamedabdallah-14/prompt-to-asset.

**Take:** we need asset-type model routing (photo vs anime vs general), and a
documented never-SUPIR-on-text policy in the router.

## 6. Video engine (`videos.py`, `cv_video.py` — ffmpeg)

**Best outside:** The ffmpeg agent skills converge on one discipline: normalize
every input (scale/pad, fps, setsar, loudnorm per-clip *before* mixing), then
`xfade`/`acrossfade`; duration-preserving transitions via tpad+apad freeze pads;
`loudnorm=I=-16:TP=-1.5:LRA=11` for social delivery; text overlays as
pre-rendered PNG `overlay` (drawtext inside xfade graphs hangs); per-clip loudnorm
not global gain; scene tools (silence.py, cut.py, look.py contact sheets).
Sources: limchinhan123/ffmpeg-vlog-pipeline SKILL.md; godot-fun video-merge-gpu
reference.md; mua47105-hue ffmpeg-command-reference; adeyholar/ffmpeg-skill;
soaing2024/video-studio pipeline.md.

**Take:** we lack `loudnorm`, watermark `overlay`, rotate/flip ops, and
per-clip normalization before concat.

## 7. Captions (`captions.py` — whisper → ASS → libass burn)

**Best outside:** Capite (MIT, 27 viral styles: Hormozi, MrBeast, Crimson Pop,
Submagic Storyteller…) and motionly's caption-animation skill define the bar:
word-level timing (never fake by splitting lines), 1–4 word pages, per-word pop/
karaoke-fill/bounce/glow motion engines, 9:16 safe-area placement, faster-whisper
for CPU-friendly word timestamps, .ass scripts + libass burn (not OpenCV
compositing). Editing a transcript must not require re-transcription.
Sources: github.com/muneebkhan08/capite; github.com/coppsary/motionly
caption-animation SKILL.md; github.com/adnanhoque369-ai/ai-video-captions.

**Take:** we have 4 styles and one karaoke fill. Gold = more viral styles
(neon glow, beast-bounce with `\t` scale tags, clean podcast top-center) and
per-word pop animation.

## 8. Color grading (`studio.py` — op_grade / FILTER_PRESETS)

**Best outside:** Pro flow = .cube LUTs (DaVinci/Premiere/OBS all consume them;
ffmpeg `lut3d=interp=tetrahedral`), lift/gamma/gain wheels, curves,
HSL qualifiers, scopes (waveform/vectorscope) — plus custom LUT import with a
strength slider, and grain via Overlay/Soft-Light at low strength (grain must
move per frame for video). Film-emulation LUTs change colour only; grain/
halation are separate passes.
Sources: isaacrowntree/color-grade-ai; twn39/color-science-skills;
pixflow.net cinematic LUT guide; dev.to .cube in Blender.

**Take:** we have 10 baked presets but no LUT import — the single biggest
grading gap. Gold = a pure-numpy .cube reader (trilinear) + `op_lut` +
ffmpeg `lut3d` video path.

## 9. NL intent (`intent.py` — deterministic parser)

**Best outside:** The NL→edit frontier is LLM-routed (the user's own standing
rule: no hardcoded intent shortcuts when a brain is available), but the
deterministic layer still matters as the offline fallback and audit surface.
The instruction-editing literature (InstructPix2Pix → MagicBrush → UltraEdit →
MGIE/plan-then-edit) shows the winning pattern: parse instruction → plan →
execute, with multi-turn sessions.

**Take:** extend pattern coverage (bg remove, upscale, cartoonize, meme,
caption, collage, watermark) and make `describe_plan` presentation god-tier
(themed output), since the plan text is what the user actually reads.

## 10. Layers (`layers.py` — LayerStack over studio renderers)

**Best outside:** Photoshop/GIMP: ~27 blend modes with exact math (multiply,
screen, overlay, soft-light Pegtop variant, hard/vivid/linear/pin light,
difference, exclusion, hue/sat/color/luminosity), per-layer opacity + masks +
transforms (move/scale/rotate), non-destructive reorder.
Sources: photoshoptrainingchannel.com blend math; docs.gimp.org layer modes;
en.wikipedia.org/wiki/Blend_modes.

**Take:** our `_blend_fn` covers a subset; layers lack transforms (move/scale/
rotate) and duplicate. Gold = layer transforms + duplicate + more blend modes.

## 11. Text-in-image editing (`edittext.py` — tesseract + inpaint)

**Best outside:** Best practice (prompt-to-asset synthesis): do NOT render brand
text in diffusion models — composite via app-layer text rendering with the
brand font. For existing text: OCR detect → LaMa-style inpaint → re-render.
Translation of in-image text is a real product feature (camera translators).

**Take:** add remove-text-only mode and multi-region replace; keep the
render-in-app-layer principle.

## 12. Transcript editing (`transcript_edit.py` — Descript-style)

**Best outside:** Descript's gold-standard flow: transcript IS the timeline —
delete text = cut media; "Shorten word gaps" collapses pauses over a threshold;
one-click filler removal; Studio Sound cleanup. Open alternatives (cut-clean,
VidClean): non-destructive toggle, jump-cut export, filler + silence in one
pass, leave a breath of surrounding silence so cuts don't sound clipped.
Sources: descript.com silence-removal guide; klipa.ai filler-word workflow;
github.com/srikant/cut-clean.

**Take:** we have filler/silence removal and topic keep. Missing: WebVTT
export, word search (text → timestamps), auto-chapters from pause structure.

## 13. Jobs (`jobs.py` — JobManager)

**Best outside:** Agent skills show the pattern: bounded worker pool, progress
callbacks, cancellation, retry, status polling — `submit`/`cancel`/`get_output`
mirrored in comfy skills.

**Take:** we lack cancel/retry. Gold = `cancel(job_id)` + `retry(job_id)`.

## 14. Presentation / style

Across every tool above, the best ones present: themed output (Capite's style
picker, color-grade-ai's report format), progress you can watch, and plans you
can read. Our `describe_plan` / `EditStudio.describe` are functional but flat.
Gold = output themes (`plain` vs `rich`) on plan + session reports.
