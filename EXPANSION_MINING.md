# Expansion Mining Report: Prompt Engine + img2img Filler + AI Editing Arsenal

Written BEFORE build commits (standing rule).

## 1. Prompt engine (the big one)

### LTX official prompt engineering (Lightricks/LTX-Video repo)
Structure, in order, single flowing paragraph <200 words, chronological:
1. Main action — one sentence, start here
2. Movement/gesture specifics
3. Appearance (precise, physical)
4. Background/environment
5. Camera angles AND movements
6. Lighting and colors
7. Changes/sudden events
Plus: `enhance_prompt=True` exists in the diffusers pipeline — but it's a black box; our engine is explicit and inspectable.

### Community gold (ComfyUI/LTX power users)
- **Temporal dynamics**: describe movement AND time or you get "living photo" syndrome.
- **Camera motion vocabulary**: dolly zoom, tracking shot, handheld shake, low-angle pan, tilt up.
- **Subject dynamics**: micro-expressions, hair flutter, rhythmic breathing — the aliveness cues.
- **Environmental motion**: mist, flickering shadows, rain, dust motes.
- **Time-stamping**: "at 0s she raises her hand; at 2s the camera pushes in" — beats, not vibes.
- **Physical over vibe**: "pores, flyaway hair, fabric weave, lens grain" beats "beautiful, epic".
- **Lens specs**: focal length, bokeh type — prevents inconsistent DoF.
- **Negative prompts**: blurry, low quality, distorted, watermark, text, logo, over/underexposed, jittery, morphing faces, unnatural motion.

### Backend shapes (different models want different prompts)
- **LTX**: chronological paragraph, action-first, <200 words.
- **Wan**: similar, tolerates more stylized language.
- **Motion/CPU fallback**: dense keyword phrases (no grammar to parse).
- **Image (SD1.5)**: comma-separated tags + weighting.

### Engine design (from mining)
Sections: action_sentence / beats[] (time-stamped choreography) / appearance /
environment / camera{angle, movement, lens} / lighting / style_tags /
physical_details / negative / consistency_anchors.
Two fill modes: LLM expansion (brain fills sections — best) and rule-based
structural assembly (offline fallback — deterministic, honest).

## 2. img2img filler

- **Missing frames**: true optical flow needs CV. Honest CPU: eased cross-dissolve morph (real in-between content for small gaps, labeled as such).
- **Extended backgrounds**: neural outpaint when available; CPU: mirror-pad + blur-blend (standard photographic extension, honest).
- **In-between content**: transitional frame synthesis between two shots.

## 3. AI editing arsenal

| Op | Neural (wired, honest) | CPU fallback (real technique) |
|---|---|---|
| inpaint | imggen/edit.py inpaint (SD) | cv_ops telea / skimage |
| outpaint | imggen/edit.py outpaint | mirror-pad + blend |
| style transfer | genedit v2v / img2img high strength | — (needs model, honest) |
| **face swap** | **inswapper_128.onnx (InsightFace)** — **works on CPU via onnxruntime!** | n/a — the model IS the CPU path |
| background replace | composite (needs mask) | feather + color-match composite (real photo technique) |
| object removal | inpaint masked region | telea |
| object addition | inpaint w/ prompt on mask | — (needs model, honest) |
| relighting | SynthLight (diffusion, GPU) | directional dodge/burn + temp shift (real photo technique) |
| expression edit | LivePortrait (GPU) | — (needs model, honest) |

### Face swap gold (inswapper)
- `https://github.com/haofanwang/inswapper` — onnxruntime, CPU-capable.
- Weights: `inswapper_128.onnx` from facefusion-assets releases.
- One call: `process(source_img, target_img)` → swapped. Optional CodeFormer restore.
- This is the highest-value wire: real face swap with NO GPU.

### Relighting gold
- SynthLight (Jan 2025, diffusion, identity-preserving) — GPU path.
- CPU: gradient dodge/burn is a real darkroom technique — honest "photographic relight".

### Trash-build discipline
- Don't claim inpainting without a mask path.
- Face-swap ethics: this is the owner's tool on their own media — the gating layer (not this module) governs use.
- Every op reports its backend honestly (`neural` vs `photo` vs `unavailable`).
