# Models & Packages — Complete Dependency Reference

Every model, package, and external dependency Devon needs, mined from the actual codebase (not guessed). Classified by where it runs.

**Legend:**
- 📱 **PHONE** — Termux/Android. Must be lightweight; metered connection matters.
- 💻 **PC** — Desktop/laptop. Can handle GB-scale models.
- 🖥️ **WORKSTATION** — GPU server/cloud. Heavy training and large models.
- ✅ Required | 🔶 Optional (feature degrades without it)

**⚠️ Metered connection warning:** Items marked 💰 are large downloads. Avoid on Airtel Nigeria mobile data unless on WiFi.

---

## 1. Python Packages

Core has **zero** mandatory dependencies (`pyproject.toml`: `dependencies = []`). Everything is optional acceleration. Install groups via `pip install nomorals[<group>]`.

### Core / Always Useful

| Package | For | Platform | Required? | Install |
|---------|-----|----------|-----------|---------|
| `numpy>=1.24` | Vector store, training, DSP | 📱💻🖥️ | 🔶 (fast group) | `pip install "nomorals[fast]"` |
| `pyyaml>=6.0` | Config files | 📱💻🖥️ | 🔶 | `pip install "nomorals[config]"` |
| `pytest>=7.4`, `ruff>=0.3` | Dev/test | 💻🖥️ | 🔶 | `pip install "nomorals[dev]"` |

### Media

| Package | For | Platform | Required? | Install |
|---------|-----|----------|-----------|---------|
| `yt-dlp>=2024.4.9` | Media download (`/play`, `/hub`) | 📱💻🖥️ | 🔶 | `pip install "nomorals[media]"` |
| `Pillow>=10.0` | Image editing | 📱💻🖥️ | 🔶 | `pip install "nomorals[media-edit]"` |
| `opencv-python-headless>=4.8` | Video/image processing | 💻🖥️ | 🔶 | `pip install "nomorals[media-edit]"` |
| `scikit-image>=0.22` | Advanced image ops | 💻🖥️ | 🔶 | `pip install "nomorals[media-edit]"` |

### Voice (TTS/STT)

| Package | For | Platform | Required? | Install |
|---------|-----|----------|-----------|---------|
| `edge-tts>=6.1` | Cloud TTS fallback | 📱💻🖥️ | 🔶 | `pip install "nomorals[voice]"` |
| `openai-whisper>=20231117` | STT (reference impl, slowest) | 💻🖥️ | 🔶 | `pip install "nomorals[voice]"` |
| `sounddevice>=0.4` | Audio I/O | 📱💻🖥️ | 🔶 | `pip install "nomorals[voice]"` |
| `faster-whisper` | STT (primary, CTranslate2) | 📱💻🖥️ | 🔶 recommended | `pip install faster-whisper` |
| `piper-tts` | On-device TTS (ONNX) | 📱 | 🔶 phone TTS | `pip install piper-tts` |
| `onnxruntime` | Piper/Chatterbox inference | 📱💻 | 🔶 | `pip install onnxruntime` |
| `chatterbox-tts` | Voice cloning TTS (MIT) | 💻🖥️ | 🔶 public voice | `pip install chatterbox-tts` |
| `TTS` (coqui) | XTTS v2 cloning (private) | 💻🖥️ | 🔶 private voice | `pip install TTS` |

### Chat Adapters

| Package | For | Platform | Required? | Install |
|---------|-----|----------|-----------|---------|
| `telethon>=1.36` | Telegram MTProto (personal account) | 📱💻🖥️ | 🔶 | `pip install "nomorals[chat]"` |
| `discord.py>=2.3` | Discord bot | 💻🖥️ | 🔶 | `pip install "nomorals[chat]"` |

### ML / Training

| Package | For | Platform | Required? | Install |
|---------|-----|----------|-----------|---------|
| `torch>=2.1` | All neural models | 💻🖥️ | 🔶 | `pip install "nomorals[train]"` |
| `transformers>=4.38` | LLM/model loading | 💻🖥️ | 🔶 | `pip install "nomorals[train]"` |
| `datasets>=2.17` | Training data | 🖥️ | 🔶 | `pip install "nomorals[train]"` |
| `peft>=0.9` | LoRA/QLoRA | 🖥️ | 🔶 | `pip install "nomorals[train]"` |
| `unsloth`, `trl>=0.8` | Fine-tuning | 🖥️ | 🔶 | `pip install "nomorals[finetune]"` |
| `huggingface-hub>=0.22` | Model downloads | 📱💻🖥️ | 🔶 | `pip install "nomorals[hub]"` |
| `diffusers>=0.27` | Image generation compat | 💻🖥️ | 🔶 | `pip install "nomorals[genimg]"` |
| `rembg>=2.0` | Background removal | 💻🖥️ | 🔶 | `pip install "nomorals[genimg]"` |
| `ultralytics>=8.0` | YOLOv8 person detection (scene intel) | 📱💻🖥️ | 🔶 | `pip install ultralytics` 💰 ~50MB + `yolov8n.pt` ~6MB auto |
| `torchreid` | OSNet person re-ID (scene intel) | 📱💻🖥️ | 🔶 | `pip install torchreid` + `osnet_x1_0` ~9MB |
| `scenedetect` | PySceneDetect adaptive segmentation | 📱💻🖥️ | 🔶 | `pip install scenedetect[opencv]` |
| `opencv-python>=4.8` | Frame analysis, tracking, face detect | 📱💻🖥️ | 🔶 | `pip install opencv-python` 💰 ~90MB |

### Other

| Package | For | Platform | Required? | Install |
|---------|-----|----------|-----------|---------|
| `pandas>=1.5`, `matplotlib>=3.7`, `scikit-learn>=1.3` | Finance/data science | 💻🖥️ | 🔶 | `pip install "nomorals[ta]"` |
| `python-docx`, `openpyxl`, `pytesseract`, `pdf2image` | Document parsing | 💻🖥️ | 🔶 | `pip install "nomorals[docs]"` |
| `boto3>=1.34`, `paramiko>=3.4` | Cloud connectors | 💻🖥️ | 🔶 | `pip install "nomorals[cloud]"` |
| `playwright>=1.42` | Browser automation | 💻🖥️ | 🔶 | `pip install "nomorals[browser]"` |

**One-liner for everything:** `pip install "nomorals[all]"`

---

## 2. TTS Voice Models

Source: `nomorals/voice/fetch.py` MODEL_REGISTRY. Fetch via `nm voice fetch --backend <name>` or auto-download on first use.

### Phone-Friendly (📱)

| Model | HF Repo | Size | License | For |
|-------|---------|------|---------|-----|
| **Piper** | `rhasspy/piper-voices` | ~60-100 MB per voice | MIT | On-device TTS, fastest |
| **Parakeet** (STT) | `istupakov/parakeet-tdt-0.6b-v3-onnx` | ~640 MB | CC-BY-4.0 | Dictation STT |
| **Fish S1 Mini** | `fishaudio/openaudio-s1-mini` | ~1 GB | Open weights | Lightweight TTS |
| **Qwen3-TTS 0.6B** | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | ~1.2 GB | Apache-2.0 | Expressive TTS |

```bash
# Piper voice (pick ONE, not the whole repo)
nm voice fetch --backend piper --voice en_US-lessac-medium
```

### PC (💻)

| Model | HF Repo | Size 💰 | License | For |
|-------|---------|---------|---------|-----|
| **Chatterbox** | `ResembleAI/chatterbox` | ~2 GB | MIT | Best free cloning TTS (public) |
| **Chatterbox Turbo** | (same repo) | ~1.5 GB | MIT | Live voice agents, streaming |
| **F5-TTS** | `SWivid/F5-TTS` | ~1-2 GB | CC-BY-NC | Highest-fidelity cloning |
| **CosyVoice 3** | `FunAudioLLM/CosyVoice-3` | 1-2 GB | MIT | Multilingual cloning |
| **OmniVoice** | `k2-fsa/OmniVoice` | ~1-2 GB | Apache-2.0 | 600+ languages |
| **XTTS v2** | `tts_models/multilingual/multi-dataset/xtts_v2` (via `TTS` package, Coqui servers) | ~2 GB | CC-BY-NC | Private voice cloning |
| **Orpheus 3B** | `canopylabs/orpheus-3b-0.1-ft` | ~6 GB fp16 | Apache-2.0 | Expressive speech LLM |

### Workstation (🖥️)

| Model | HF Repo | Size 💰 | License | For |
|-------|---------|---------|---------|-----|
| **Fish S2 Pro** | `fishaudio/s2-pro` | ~8 GB | Research only | SOTA open TTS |
| **Dia 1.6B** | `nari-labs/Dia-1.6B-0626` | ~3.5 GB | Apache-2.0 | Two-speaker dialogue (GPU only) |
| **Qwen3-TTS 1.7B** | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | ~3.5 GB | Apache-2.0 | Larger expressive TTS |

---

## 3. STT (Speech-to-Text) Models

Source: `nomorals/voice/stt.py`. Priority: faster-whisper → parakeet → openai-whisper.

| Model | Size | Platform | Install |
|-------|------|----------|---------|
| **faster-whisper** `tiny` | ~75 MB | 📱 | `pip install faster-whisper` (auto-downloads) |
| **faster-whisper** `base` | ~145 MB | 📱💻 | (same) |
| **faster-whisper** `small` | ~460 MB | 💻 | (same) |
| **faster-whisper** `large-v3-turbo` | ~800 MB | 💻 | (same, recommended) |
| **Parakeet TDT 0.6B** | ~640 MB | 📱💻 | Auto via `onnx-asr` |
| **openai-whisper** | varies | 💻🖥️ | `pip install openai-whisper` (slowest, reference only) |

---

## 4. Music / Vocal Models

Source: `nomorals/media/vocals.py`, `nomorals/media/ace_step.py`.

| Model | For | Size 💰 | Platform | Install |
|-------|-----|---------|----------|---------|
| **ACE-Step** | Instrumental bed generation | ~4.8 GB (turbo DiT) + VAE + LM 💰 | 💻🖥️ | `git clone https://github.com/ace-step/ACE-Step-1.5 && cd ACE-Step-1.5 && uv sync` (needs Python 3.10+, NVIDIA GPU, CUDA) |
| **DiffSinger** | Singing voice synthesis | ~1-2 GB + acoustic model | 💻🖥️ | `git clone https://github.com/openvpi/DiffSinger` (not pip-installable) |
| **RVC** | Voice conversion for singing | Model-dependent | 💻🖥️ | `pip install rvc-python` + trained `.pth` + `.index` per voice |
| **Demucs** | Stem separation (vocals/drums/bass) | ~500 MB | 💻🖥️ | `pip install demucs` |

**Phone fallback:** TTS vocals via `vocal_lite.py` (uses any installed TTS backend — no separate model needed).

---

## 4b. Directed Animation + Lip Sync Models

Source: `nomorals/media/directed/`.

| Model | For | Size 💰 | Platform | Install |
|-------|-----|---------|----------|---------|
| **MimicMotion_1-1.pth** + DWPose onnx | Pose-guided video (photo + pose → directed motion) | ~5 GB + ~200 MB 💰 | 🖥️ (16GB VRAM) | `git clone https://github.com/Tencent/MimicMotion ~/.devon-models/mimicmotion/repo` + download weights into `~/.devon-models/mimicmotion/` |
| **Wav2Lip** (wav2lip.pth + s3fd.pth) | Video lip sync | ~960 MB 💰 | 💻🖥️ (8GB VRAM) | `git clone https://github.com/zdh6090/Wav2Lip ~/.devon-models/wav2lip/repo` + weights. ⚠️ Research/non-commercial license |
| **LatentSync 1.5** (latentsync_unet.pt + whisper tiny.pt) | Higher-quality lip sync | ~2-3 GB 💰 | 💻🖥️ (8GB VRAM; v1.6 needs 18GB) | `git clone https://github.com/bytedance/LatentSync ~/.devon-models/latentsync/repo` + `source setup_env.sh`. Apache 2.0 |
| **SadTalker** | Photo → talking head | ~2 GB + checkpoints 💰 | 💻🖥️ | `git clone https://github.com/OpenTalker/SadTalker ~/.devon-models/sadtaker/repo` + checkpoint script |

**CPU fallbacks (no model):** pose-guided mesh warp (directed motion) + audio-envelope jaw warp (lip sync). Honest 2D, labeled `warp`.

---

## 5. Image Models

Source: `nomorals/media/imggen/`. Devon's own UNet/DDPM implementation; can load open SD1.5 weights.

| Model | For | Size 💰 | Platform | Source |
|-------|-----|---------|----------|--------|
| **SD1.5** (any checkpoint) | Text-to-image via Devon's loader | ~4 GB | 💻🖥️ | HuggingFace (e.g. `runwayml/stable-diffusion-v1-5`) |
| **Devon's own** | Trained via `nomorals/media/imggen/train.py` | Varies | 🖥️ | Train locally |

---

## 6. LLM / Brain Models

| Model | For | Size 💰 | Platform | Source |
|-------|-----|---------|----------|--------|
| **codebeast-3.8b** (Q4_K_M GGUF) | Phone brain | ~2.3 GB | 📱 | `Cutyp/codebeast-3.8b` (private HF) |
| **codebeast-7b-vl** | Vision + heavy tasks | ~16 GB fp16 | 🖥️ | `Cutyp/codebeast-7b-vl` (private HF, training) |
| **Qwen2.5-VL-7B-Instruct** | 7B training base | ~16 GB | 🖥️ | `Qwen/Qwen2.5-VL-7B-Instruct` |

---

## 7. System Packages

| Package | For | Platform | Install |
|---------|-----|----------|---------|
| **ffmpeg** | Audio/video conversion, voice notes | 📱💻🖥️ | Termux: `pkg install ffmpeg`<br>Debian: `sudo apt install ffmpeg`<br>Mac: `brew install ffmpeg` |
| **espeak-ng** | System TTS fallback, Piper phonemizer | 📱💻 | Termux: `pkg install espeak-ng`<br>Debian: `sudo apt install espeak-ng` |
| **Node.js ≥18** | WhatsApp bridge | 📱💻🖥️ | Termux: `pkg install nodejs`<br>Or: [nodejs.org](https://nodejs.org) |
| **tesseract** | OCR (docs group) | 💻🖥️ | `sudo apt install tesseract-ocr` |

---

## 8. Node Packages (WhatsApp Bridge)

Source: `bridge/package.json`. Install: `cd bridge && npm install`.

| Package | Version | For | Size |
|---------|---------|-----|------|
| `@whiskeysockets/baileys` | ^6.7.18 | WhatsApp protocol | ~50 MB |
| `qrcode-terminal` | ^0.12.0 | QR login display | ~1 MB |
| `werift` | 0.25.0 | WebRTC for call research (spike only) | ~20 MB |

```bash
cd ~/workspace/devon/bridge && npm install
# For call research only:
npm install werift@0.25.0
```

---

## 9. Runtime-Fetched (Not Install-Time)

These download automatically on first use. Watch data usage.

| What | When | Approx Size | Where Cached |
|------|------|-------------|--------------|
| Piper voices | `nm voice fetch` | 60-100 MB each | `PIPER_VOICES_DIR` |
| Chatterbox weights | First TTS use | ~2 GB 💰 | HF cache |
| faster-whisper models | First STT use | 75 MB - 800 MB | HF cache |
| Wisdom corpus (47 texts) | `/wisdom seed` | ~20-50 MB | Wisdom corpus dir |
| RVC voice models | `/music voices add` | Varies (.pth + .index) | Voices dir |
| Training datasets | `nm train` / dataset fetch | Varies (GB) | Training data dir |

---

## Quick-Start by Platform

### 📱 Phone (Termux) — Minimal
```bash
pkg install python ffmpeg espeak-ng nodejs
pip install "nomorals[fast,media,voice,chat,config]"
pip install faster-whisper piper-tts
cd ~/workspace/devon/bridge && npm install
nm voice fetch --backend piper --voice en_US-lessac-medium
```

### 💻 PC — Full Capability
```bash
pip install "nomorals[all]"
pip install faster-whisper chatterbox-tts TTS rvc-python
# + ffmpeg, espeak-ng via system package manager
cd ~/workspace/devon/bridge && npm install
```

### 🖥️ Workstation — Training + Heavy Models
```bash
pip install "nomorals[all]"
# + GPU torch, unsloth, diffusers per above
# + large TTS models (Fish S2 Pro, Dia) as needed
```

---

*Generated 2026-10-09 from codebase mining. UNVERIFIED items are marked inline. Sizes are approximate — check before downloading on metered connections.*
