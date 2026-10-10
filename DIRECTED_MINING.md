# Directed Video Animation — Mining Report

Written BEFORE any build commits (standing rule: mine first, no rushed works).
Task: pose/motion control ("make the person raise two fingers") + camera emulation ("shot on phone" look).

## 1. Pose-guided video generation (the real models)

### MimicMotion (Tencent, ICML 2025) — PRIMARY
- Repo: https://github.com/Tencent/MimicMotion
- What: reference image + pose video → photoreal video of that person doing that motion.
- Gold: **confidence-aware pose guidance** (high-confidence keypoints weighted more — fixes hand/face distortion, exactly our "two fingers" problem), **regional loss amplification** on low-confidence regions, **progressive latent fusion** for arbitrary-length smooth video.
- Weights layout: `models/DWPose/dw-ll_ucoco_384.onnx`, `models/DWPose/yolox_l.onnx`, `models/MimicMotion_1-1.pth`.
- Inference: `python inference.py --inference_config configs/test.yaml`.
- Cost: 72-frame model needs **16GB VRAM**, ~20 min on a 4090 for a 35s demo. 16-frame U-Net minimum 8GB; VAE decoder wants 16GB (can run on CPU).
- Verdict: workstation-class. The pose input is a rendered skeleton video (DWPose wholebody), which means **we can synthesize pose videos programmatically** — we don't need a template motion clip, we can author keypoint trajectories directly.

### MagicAnimate / AnimateAnyone (MooreThreads open impl) — SECONDARY
- ReferenceNet architecture: best-in-class identity preservation (~4fps on GPU).
- Input: OpenPose 25-keypoint JSON + rendered skeleton; MediaPipe 33 → OpenPose 25 mapping needed.
- Weight trap documented in the wild: README URLs under `Moore-AnimateAnyone` are wrong; real weights at `patrolli/AnimateAnyone` + `lambdalabs/sd-image-variations-diffusers`. (Trash-build lesson: verify weight URLs against the binary, not the README.)
- Verdict: alternative backend if MimicMotion fails on a machine. Same pose-video interface.

### PoseAnything (arXiv Dec 2025) — WATCH
- Universal skeletons (non-human subjects). Not needed now; note for the future.

### LTX sign-language workflow (ComfyUI, community) — VALIDATES OUR APPROACH
- Reference image (identity) + motion video → LTX 2.3 transfers motion. ~80% hand accuracy; finger precision is the known weak point everywhere.
- Gold: proves the "template/parametric motion + identity image" pattern works on LTX-class backbones. Our LTX backend already exists in videogen/.

### GestureGAN — REJECTED for video
- Image-to-image hand gesture translation only (ACM MM 2018). No temporal model. Noted, not used.

## 2. Pose sequences: where the "direction" comes from

The models above consume pose VIDEOS. The direction layer must produce them from an action description. Options mined:

1. **Programmatic keypoint trajectories (CHOSEN):** define actions as parametric functions over a body+hand keypoint rig (COCO-18 body + 21-pt hands, DWPose-compatible layout). "Raise two fingers" = arm keypoints arc upward + hand keypoints morph fist→✌️ over N frames. Deterministic, editable, no model needed, works offline. This is the honest core: the ACTION is authored, the MODEL renders it photoreal.
2. **Template motion extraction:** DWPose on a reference clip of someone doing the action → retarget keypoints. Needs a template library; good fallback when the parametric rig lacks an action.
3. **LLM-authored trajectories:** the brain writes keyframe JSON for novel actions ("do a backflip while waving") — the rig interpolates. Future, via spine.

Gold from trash builds: recalculation beats templates — parametric beats a frozen template library for coverage.

## 3. Camera emulation (post-process, no model needed)

- **Handheld shake:** Perlin noise on 6DoF (x/y translation, roll, scale-breathing) is the industry-standard synthesis (game engines, Blender f-curve noise modifiers). Frequency ~0.5–3 Hz for idle handheld, amplitude in pixels.
- **Rolling shutter:** row-by-row time offset — implement as vertical shear proportional to horizontal shake velocity. The phone signature.
- **Phone ISP look:** over-sharpen (unsharp mask), HDR-ish local contrast, slight chroma noise grain, front-cam = vertical punch-in + wider FOV feel + softer detail.
- **Other cameras:** CCTV (interlace lines + noise + timestamp burn-in + crushed blacks), dashcam (barrel distortion + timestamp + speed overlay), cinema (2.39:1 letterbox + 24fps cadence + gentle S-curve grade).
- All implementable with PIL/numpy + ffmpeg. Zero model cost.

## 4. Architecture decision (from mining)

```
action text → pose program (parametric rig) → pose video (rendered skeleton)
                                                    ↓
reference image ──────────────────────────→ MimicMotion / AnimateAnyone → raw clip
                                                    ↓ (or CPU fallback)
                                              mesh-warp animator (PIL MESH, pose-guided)
                                                    ↓
                                              camera emulation (shake/phone/cctv/…)
                                                    ↓
                                              edit timeline (genedit pattern)
```

- Neural path: workstation (16GB VRAM). Honest `ModelUnavailable` with exact install otherwise.
- CPU path: pose-guided mesh warp — REAL directed motion (arm region rises, hand morphs), crude but honest. Labeled `warp`, never claimed as neural.
- Camera emulation is orthogonal: applies to generated OR real footage.

## 5. Models to wire NOW (not later)

| Model | Size | For | Install |
|---|---|---|---|
| MimicMotion_1-1.pth | ~5GB (fp16 UNet) | pose→video render | `pip install -r` from repo + HF weights |
| dw-ll_ucoco_384.onnx + yolox_l.onnx | ~200MB | DWPose keypoint extraction (template path) | ships with repo |
| (optional) AnimateAnyone weights | ~7GB | alt backend | `patrolli/AnimateAnyone` |

## 6. Trash-build discipline (kept)

- Verify weight URLs against the loader, not the README.
- Render engine first, UI second.
- Parametric recalculation beats frozen templates.
- Every fallback labeled honestly (`warp` ≠ `neural`).
