# Lip Sync — Mining Report

Written BEFORE build commits (standing rule: mine first, no rushed works).

## 1. The models

### Wav2Lip (zdh6090/Wav2Lip, ACM MM 2020) — PRAGMATIC PICK
- What: video + audio → lip-synced video. GAN, renders mouth at 96×96, pastes back.
- Cost: ~2GB VRAM at inference, runs on 8GB GPUs, ~25fps on Apple Silicon MPS.
- Weights: wav2lip.pth (~350MB) + wav2lip_gan.pth + s3fd.pth face detector (~100MB) + expert discriminator. ~960MB total.
- Install: `git clone https://github.com/zdh6090/Wav2Lip`, pip requirements, download weights.
- Inference: `python inference.py --checkpoint_path wav2lip.pth --face video.mp4 --audio audio.wav --outfile out.mp4`.
- Trade-off: mouth slightly soft, faint seam at the paste boundary. At small display sizes barely visible.
- License: **research / non-commercial** — commercial use prohibited. Must surface this.
- Verdict: primary backend. Fits the widest hardware.

### LatentSync 1.5/1.6 (ByteDance, Dec 2024) — QUALITY PICK
- What: audio-conditioned latent diffusion, Whisper audio embeddings, SyncNet supervision. Sharper 256–512px face region, better temporal stability than Wav2Lip.
- Cost: v1.5 = 8GB VRAM; v1.6 (512px, sharper) = 18GB VRAM.
- Weights: latentsync_unet.pt + whisper tiny.pt. Checkpoints dir layout documented in repo.
- Install: `git clone https://github.com/bytedance/LatentSync`, `source setup_env.sh`.
- License: **Apache 2.0** — commercial OK. This matters.
- Verdict: quality backend when VRAM allows. v1.5 primary alt, v1.6 for 18GB+ machines.

### SadTalker (OpenTalker, 2023) — PHOTO PICK
- What: single photo + audio → talking head with full head motion via 3DMM (3D morphable model). Audio → 3DMM expression parameters.
- Strength: alive head motion (tilts, nods) — still heads look dead on long clips.
- Weakness: lip precision worse than Wav2Lip on fast speech; slower; 3DMM rendering limits resolution; temporal artifacts on transitions.
- Verdict: photo-input backend. Complements (not replaces) video lip sync.

### MuseTalk — REJECTED
- Painful pinned-2024 install, degrades facial features per field reports. Noted, not wired.

### sync-3 / HeyGen / commercial APIs — NOTED
- Hosted, per-second pricing. Not wired (no-cloud-dependency principle for core paths), but the architecture allows an API backend later.

## 2. CPU fallback: audio-envelope jaw warp

No model, no torch, no face detector on CPU-only machines. The honest fallback:
- Extract audio → RMS energy per video frame → 95th-percentile normalize → envelope follower (fast attack, slow release) → silence gate.
- Mouth region (lower third of face box) stretches vertically with openness.
- REAL audio-driven motion: mouth opens on speech, closes on silence. Crude 2D warp, labeled `warp`, never claimed as neural.
- Face box required (no detector on CPU). Pose-track-derived when lip-syncing directed-animation output (we know where the nose is).

## 3. Dubbing pipeline

video + translated_text + voice → TTS (catalogue voice) → lip sync → mux.
Translation itself is an LLM task (no offline translator wired — honest); the brain translates, the pipeline dubs.

## 4. Architecture

```
video + audio ──┬──> LatentSync (8GB+, best quality)
                ├──> Wav2Lip (8GB, pragmatic)
                └──> envelope warp (CPU, face_box required)
photo + audio ──> SadTalker (talking head)
dub: text + voice ──> TTS ──> lip_sync ──> mux
```

## 5. Trash-build discipline

- License check first: Wav2Lip is non-commercial — surfaced in status, not buried.
- VRAM honesty: v1.6 needs 18GB, not "a GPU".
- The seam: Wav2Lip's 96px paste boundary is real — documented, not hidden.
