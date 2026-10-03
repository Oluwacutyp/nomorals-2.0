# Local (Free) Image & Video Generation — Research

> Researched 2026-10-03 for the Devon AI project (mediadeps-1.0).
> Goal: image/video generation that contends with Grok Imagine, Gemini,
> Leonardo AI, Google Flow and Nano Banana — **free by default** (local
> models, zero API keys).
>
> Standing rule applied throughout: **best free option wins, judged on
> actual quality — not popularity.** Every pick below is open-weights,
> runs locally with no key, and the default stack deliberately avoids
> anything that needs ≥16 GB VRAM.

---

## 1. Current state (gap analysis)

`nomorals/media_edit/generate.py` already has the right *architecture* —
pluggable `GenerativeBackend` interface with `edit / generate / img2img /
inpaint`, style/aspect/quality presets, and backends resolved by
`MEDIA_GEN_BACKEND` (`auto` prefers local diffusers, falls back to HF
serverless). The problems are the **defaults and the missing capability**:

| Current | Gap |
|---|---|
| Default local model: `timbrooks/instruct-pix2pix` | 2023-era editing model, 512px, dated quality — and it's used for *text-to-image* too, where it's the wrong tool |
| `HFInferenceBackend` (FLUX.1-Kontext-dev via serverless) | Not free-local: burns the $0.10/month HF free tier; cloud, not local |
| Text-to-image via `AutoPipelineForText2Image` on the editing model | No native T2I pipeline configured at all |
| Inpainting via generic auto-pipeline | Works, but untested against modern dedicated pipelines |
| `op_upscale` | Lanczos only — no real super-resolution backend wired |
| Video generation | **Completely absent.** `videos.py`/`cv_video.py` are ffmpeg/OpenCV *editing* only (trims, filters, concat, overlays) — nothing generates pixels from a prompt |

**So the build work is:** swap in modern default models (below),
add native pipeline classes per model family, and add a video backend
(`nomorals/media_edit/video_gen.py`) — the architecture in
`generate.py` already supports the extension.

---

## 2. Image generation

### 2.1 Candidates evaluated

| Model | Maker / HF repo | Params | License | VRAM (practical floor) | Steps | Native res | Verdict |
|---|---|---|---|---|---|---|
| **Z-Image-Turbo** | `Tongyi-MAI/Z-Image-Turbo` | 6B | **Apache-2.0** | 12–16 GB (bf16); ~8 GB (FP8, `unsloth/Z-Image-Turbo-FP8`) | **8** | 1024px | ⭐ **PICK — best free T2I, 2026.** #1 open-weights on the Artificial Analysis Text-to-Image Arena; beats SDXL/FLUX.1-schnell on photorealism and prompt adherence; bilingual (EN/中文) text rendering; diffusers-native (`ZImagePipeline`, `ZImageImg2ImgPipeline`, `ZImageInpaintPipeline`) |
| Z-Image-Edit | Tongyi-MAI Z-Image family (edit variant) | 6B | Apache-2.0 | same as above | 8 | 1024px | ⭐ **PICK for instruction editing** — fine-tuned for natural-language edits; the correct replacement for `instruct-pix2pix`. (Verify exact HF repo id at implementation — family confirmed, repo naming was in flux) |
| FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` | 12B | **Apache-2.0** | 24 GB (bf16); 12 GB (FP8); 8 GB (GGUF Q4_K_S, ~12 GB total download with encoders+VAE) | 4 | 1024px | Runner-up. Great prompt adherence, fully commercial-safe license, but needs quantization to run on consumer cards — heavier and slower than Z-Image-Turbo on the same GPU |
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | 12B | **Non-commercial** | 16 GB (FP8); 24–32 GB (bf16); 8 GB (GGUF Q4) | 28–50 | 1024px | Best quality-per-parameter of the 2024 generation, but the non-commercial license and VRAM floor disqualify it as *default* — keep as opt-in for quality runs |
| FLUX.2 [klein] 4B | Black Forest Labs | 4B | **Commercial BFL license** | 12–16 GB | 4 | 1024px | Fast and good, but the license isn't Apache — excluded by the "free, no strings" rule |
| SDXL 1.0 | `stabilityai/stable-diffusion-xl-base-1.0` | 3.5B | Community/OpenRAIL++ | **8 GB** (fp16, ~7 GB download incl. VAE) | 30–40 | 1024px | The ecosystem king: largest LoRA/style checkpoint zoo, best for anime and stylized work. Older, worse prompt adherence than the picks above — **keep as fallback, not default** |
| SDXL Turbo | `stabilityai/sdxl-turbo` | 3.5B | Stability community | 8 GB | 4 | 512px | Fast drafts; quality lags SDXL proper. Obsolete as a default now that Z-Image-Turbo exists |
| SSD-1B | `segmind/SSD-1B` | 1.3B | OpenRAIL | ~6 GB | 4 | 1024px | Distilled SDXL; fine, but strictly worse than SDXL proper — no reason to pick in 2026 |
| SD 1.5 | `runwayml/stable-diffusion-v1-5` | 0.9B | OpenRAIL-M (ungated) | **4 GB** (~3.4 GB download) | 20–30 | 512px | Legacy-tier only: GPUs with ≤4 GB VRAM. Enormous checkpoint ecosystem, but quality is a generation behind |
| Qwen-Image | `Qwen/Qwen-Image` | 20B | Apache-2.0 | 24 GB+ | 50 | 1328px | **Text-rendering champion** — nothing local renders posters/signage text better. 20B = not a default; wire as an opt-in specialist for the "readable text" case |
| PixArt-Sigma / Kandinsky / Würstchen / Lumina | various | — | various | — | — | — | Evaluated and **rejected**: Kandinsky and Würstchen are 2023-era and visibly behind; PixArt-Sigma is decent but has a thin ecosystem; Lumina-Image 2.0 is interesting but harder to deploy than the picks. Popularity ≠ quality, but here the quality leaders also happen to win on ease |

**Why Z-Image-Turbo over the popular picks (SDXL / FLUX.1-schnell):**
- 8 steps, no CFG (`guidance_scale=0.0`) → sub-second on datacenter GPUs,
  a few seconds on a consumer card — faster than SDXL's 30 steps and
  FLUX-dev's 50, at higher benchmarked quality than both.
- Apache-2.0 (commercial-safe, no gating on the Turbo repo), single
  download, diffusers-first-class (txt2img / img2img / inpaint pipelines
  exist).
- #1 open-weights model on the Artificial Analysis T2I Arena in 2026 —
  the "best free, not most popular" pick by the numbers.
- Z-Image-Edit covers the instruction-editing lane with the same
  weights/family, so one model family serves T2I + I2I + editing.

### 2.2 Image install

```bash
# core stack (CUDA 12.x)
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install diffusers transformers accelerate safetensors huggingface_hub pillow

# download (weights ~12 GB bf16; ~6.5 GB for the unsloth FP8 build)
hf download Tongyi-MAI/Z-Image-Turbo                      # default
# or low-VRAM: hf download unsloth/Z-Image-Turbo-FP8
```

Minimal inference (verified pattern from diffusers docs + community):

```python
import torch
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16).to("cuda")
image = pipe(prompt="...", height=1024, width=1024,
             num_inference_steps=8, guidance_scale=0.0).images[0]
```

`ZImageImg2ImgPipeline` and `ZImageInpaintPipeline` take the same
checkpoint — one download covers all four ops in `generate.py`.

---

## 3. Video generation

### 3.1 Candidates evaluated

| Model | Maker / HF repo | Params | License | VRAM (practical floor) | Clip | Speed (RTX 4090-class) | Verdict |
|---|---|---|---|---|---|---|---|
| **LTX-Video 2B distilled** (`ltxv-2b-0.9.8-distilled`) | `Lightricks/LTX-Video` | 2B | **OpenRAIL-M (commercial OK)** | **8 GB** | 5 s, 512×320 → 720×480 | ~10 s | ⭐ **PICK — default video.** Fastest by far (~15× faster than the 13B), diffusers-native (`LTXPipeline`, `LTXImageToVideoPipeline`), one `pip install`, no exotic deps |
| **Wan2.1-T2V-1.3B** | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 1.3B | **Apache-2.0** | **8.19 GB** | 5 s, 480p | ~4 min | ⭐ **PICK — quality tier.** Best motion/physics coherence of any ≤8 GB model; diffusers `WanPipeline`. Slower than LTX-2B but visibly better-looking video |
| LTX-2 (19B / 22B / 2.3 / 2.5) | `Lightricks/LTX-2` | 19–22B | gated, commercial terms | 8 GB (GGUF Q3+heavy offload) – 24 GB (bf16) | 10–20 s, up to 4K, **native audio** | minutes | #1 on open video leaderboards, but ~25–66 GB download (GGUF + video VAE + audio VAE + Gemma encoder) and a gated repo. **Watch, don't default** — revisit if Devon grows a ComfyUI-style GGUF path |
| Wan2.2-T2V-5B / -A14B | Alibaba | 5B / 14B-MoE | Apache-2.0 | 10 GB / 16–24 GB | 720p | minutes | Benchmark leader for T2V, but the 1.3B gets you 80% of the quality at half the VRAM |
| CogVideoX-2b / -5b | `THUDM/CogVideoX-2b` | 2B / 5B | Apache-2.0 | 8–10 GB / 16 GB | 6 s, 720×480 | minutes | Solid but superseded: Wan2.1-1.3B beats the 2B at the same VRAM; the 5B needs 16 GB |
| HunyuanVideo | `hunyuanvideo-community/HunyuanVideo` | 13B | Community (gated) | 40–80 GB nominal; ~12–16 GB via GGUF+tiling+offload | 10 s, 720p | slow | Quality ceiling for open video, but the VRAM games needed disqualify it as default |
| Mochi-1 | genmo/mochi-1-preview | 10B | Apache-2.0 | 24 GB | 5–6 s, 848×480 | slow | Heavy for what it delivers; outclassed by Wan2.2 |
| Stable Video Diffusion | `stabilityai/stable-video-diffusion-img2vid-xt` | 1.5B | Community | 16 GB | image→video only | — | Dated; LTX's `LTXImageToVideoPipeline` does I2V better, faster, lighter |
| AnimateDiff | `guoyww/animatediff-motion-adapter-v1-5-2` + SD1.5 | — | Apache-2.0 | **6 GB** | 1–2 s | fast | Legacy-tier only: 4–6 GB GPUs. Short, low-fidelity clips |
| Open-Sora | hpcaitech/Open-Sora | — | Apache-2.0 | 24 GB+ | variable | slow | Research-grade; not competitive in 2026 |

**Why this two-tier pick (LTX-2B distilled default, Wan2.1-1.3B quality):**
- Both run on the same 8 GB consumer GPU — no hardware fork in the
  default stack, just a speed/quality knob.
- Both are diffusers pipelines: `LTXPipeline` / `LTXImageToVideoPipeline`
  and `WanPipeline` — consistent with the existing
  `MediaEditBackend` pattern, no ComfyUI or custom repos required.
- LTX-2B-distilled is genuinely real-time-class (~10 s per 5 s clip on a
  4090, much faster than anything else at 8 GB), which matters for a
  chat-driven agent where the user waits.
- Wan2.1-1.3B is Apache-2.0 (fully free), and its motion coherence is the
  closest any 8 GB model gets to the paid tools.

### 3.2 Video install

```bash
# same core stack as image; add:
pip install imageio imageio-ffmpeg   # mp4 export

# downloads: LTX-2B distilled is a few GB inside the Lightricks repo;
# Wan2.1-1.3B-Diffusers is ~8-10 GB (transformer + umt5 encoder + VAE)
hf download Lightricks/LTX-Video
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

Minimal inference (from the official Lightricks diffusers docs):

```python
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video", torch_dtype=torch.bfloat16).to("cuda")
frames = pipe(prompt="...", width=704, height=480,
              num_frames=161, num_inference_steps=8).frames[0]
export_to_video(frames, "out.mp4", fps=24)
```

Use `LTXImageToVideoPipeline` for image-to-video, and `WanPipeline`
from `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` for the quality tier.
`export_to_video` needs `imageio-ffmpeg` — **ffmpeg must be installed**,
which aligns with the existing `videos.py` requirement.

---

## 4. Recommended default stack ("what Devon should use out of the box")

| Role | Model | Repo | VRAM | Why |
|---|---|---|---|---|
| Text-to-image | **Z-Image-Turbo** | `Tongyi-MAI/Z-Image-Turbo` | 12–16 GB bf16 · ~8 GB FP8 | Best free quality, 8 steps, Apache-2.0 |
| img2img / inpaint | **Z-Image-Turbo** | same (pipelines: `ZImageImg2ImgPipeline`, `ZImageInpaintPipeline`) | same | One download, all image ops |
| Instruction edit | **Z-Image-Edit** | Tongyi-MAI Z-Image family (verify repo id) | same | Replaces `instruct-pix2pix` with 2026 quality |
| Video (fast) | **LTX-Video 2B distilled** | `Lightricks/LTX-Video` | 8 GB | ~10 s per 5 s clip, diffusers-native |
| Video (quality) | **Wan2.1-T2V-1.3B** | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 8 GB | Best-looking ≤8 GB video, Apache-2.0 |
| Text-in-image specialist (opt-in) | Qwen-Image | `Qwen/Qwen-Image` | 24 GB+ | Posters/signage with readable text |
| Low-VRAM fallback (8 GB, images) | SDXL 1.0 | `stabilityai/stable-diffusion-xl-base-1.0` | 8 GB | Ecosystem/Loras when Z-Image FP8 still too heavy |
| Legacy fallback (4–6 GB, images/video) | SD 1.5 / AnimateDiff | `runwayml/stable-diffusion-v1-5` | 4 GB | Old GPUs only |
| CPU-only (images) | SDXS-512-0.9 via OpenVINO, or FastSD CPU app | `optimum[openvino]` | 0 GPU | See §5 |

Env-var mapping for the implementation phase (extends existing
conventions in `generate.py`):

```
MEDIA_GEN_BACKEND=diffusers            # unchanged
MEDIA_GEN_DIFFUSERS_MODEL=Tongyi-MAI/Z-Image-Turbo     # replaces instruct-pix2pix
MEDIA_GEN_EDIT_MODEL=<Z-Image-Edit repo id>            # instruction edits
MEDIA_VIDEO_BACKEND=diffusers                          # new: video side
MEDIA_VIDEO_MODEL=Lightricks/LTX-Video                 # new, default (fast tier)
MEDIA_VIDEO_QUALITY_MODEL=Wan-AI/Wan2.1-T2V-1.3B-Diffusers  # new, quality tier
```

Suggested implementation shape: a new `nomorals/media_edit/video_gen.py`
mirroring `generate.py`'s backend interface
(`generate(prompt, ...) -> mp4 path`, `img2video(image, prompt, ...)`),
so the existing studio/job/CLI plumbing reuses the same patterns.

---

## 5. CPU-only users (no GPU)

Be honest: **diffusion on CPU is slow, and video on CPU is not
practical.** Options, ranked:

1. **FastSD CPU** (`github.com/rupeshs/fastsdcpu`) — the best CPU image
   story in 2026. OpenVINO backend: **0.82 s per 512×512 image**
   (SDXS-512-0.9) on an i7-12700; as of 2026-06 it also runs
   **FLUX.2-klein-4B in 2–4 steps via OpenVINO**. It's a GUI/CLI app,
   not a Python library — recommend it as the external fallback, not a
   Devon dependency.
2. **`optimum[openvino]`** — Python-native: `OVStableDiffusionPipeline`
   (SD 1.5 / SDXS) runs in seconds per image on a modern Intel CPU.
   This is the implementable CPU path for Devon (`pip install
   optimum[openvino]`).
3. **Z-Image-Turbo on CPU via diffusers** — works (bf16, ~30 s for
   512×512/8 steps on a server-class CPU) if the user has ≥16 GB RAM;
   fine as a last-resort "same model, slower" mode.
4. **Video on CPU: don't.** No practical local CPU video model exists;
   LTX-2B-distilled *loads* on CPU but a 5 s clip takes far too long to
   be usable. Document the limitation; offer queue-and-wait or the
   HF-serverless fallback instead of pretending otherwise.

---

## 6. Honest limits of this research

- VRAM figures are practical floors gathered from 2026 community
  reports and model cards, not lab measurements on Devon's hardware —
  **the implementation phase must smoke-test the default stack on a
  real 8 GB card** before claiming it as default.
- Exact download sizes marked `~` should be confirmed from the HF repo
  at implementation time.
- Z-Image-Edit's exact HF repo id needs verification (the family and
  Apache-2.0 license are confirmed; naming was in flux at research
  time).
- LTX-2 (19B) is the quality ceiling and worth revisiting once Devon
  has a GGUF/offload path — it's deliberately excluded from the
  default because its full setup is a ~25–66 GB gated download.

## 7. Sources

- Model cards: `Tongyi-MAI/Z-Image-Turbo` (Apache-2.0),
  `unsloth/Z-Image-Turbo-FP8`, `Lightricks/LTX-Video`,
  `Lightricks/LTX-Video-0.9.7-distilled`, `Lightricks/LTX-2`,
  `black-forest-labs/FLUX.1-schnell`
- diffusers docs: `ZImagePipeline` / `ZImageImg2ImgPipeline` /
  `ZImageInpaintPipeline`, `LTXPipeline` / `LTXImageToVideoPipeline`
- 2026 roundups: localaimaster.com (best local image models; FLUX
  GGUF vs FP8 VRAM table), intelligibberish.com (ComfyUI VRAM table),
  medium.com/@jon_davis (open-source video stack 2026), siliconflow.com
  (top open T2V models 2026), Anil-matcha/awesome-ai-image-models
- FastSD CPU: github.com/rupeshs/fastsdcpu (OpenVINO CPU benchmarks,
  2026-06 FLUX.2-klein-4B support)
