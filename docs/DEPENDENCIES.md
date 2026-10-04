# Devon AI — Dependencies

> **Philosophy: Devon just works.** The core package has zero mandatory
> third-party dependencies (`pyproject.toml` declares `dependencies = []`),
> but every capability Devon offers has a home in a pip extra or a
> documented install command below. **Install the extra for what you use —
> don't chase "pip install X" hints.** If you want everything, one command
> does it:
>
> ```bash
> pip install .[all]        # from the repo root — every extra, everything works
> ```
>
> Check what's live on your machine any time with:
>
> ```bash
> nm doctor                # feature report: which optional deps are present
> ```

Verified against the actual code on branch `main` (AST scan of all
628 files under `nomorals/`, plus `pyproject.toml`, the `compat.py` feature
table, and the `bridge/package.json`). Every pip package below is imported
somewhere in `nomorals/` — nothing is listed on guesswork.

## Quick install

| Setup | Command |
|---|---|
| Core only (agents, missions, memory, CLI — stdlib only) | `pip install .` |
| **Everything (recommended)** | `pip install .[all]` |
| Standard (fast + media + editing + Hub) | `pip install .[fast,media,media-edit,hub]` |
| Media-heavy (download + edit + OCR) | `pip install .[media,media-edit]` + system: `ffmpeg`, `tesseract-ocr`, `poppler-utils` (see System dependencies) |
| Training (fine-tune on GPU) | `pip install .[train,finetune]` |
| Development | `pip install .[dev]` |

After the pip step, add the **system binaries** you need (ffmpeg,
tesseract, …) — those aren't pip packages, so they still need
`apt`/`brew` (see System dependencies). Then set any **API keys** your
features need (see API keys / env vars).

## Extras reference

Exactly what's in `pyproject.toml` (`[project.optional-dependencies]`):

| Extra | Packages | Unlocks |
|---|---|---|
| `fast` | `numpy>=1.24` | Accelerated vector similarity, the `nomorals/ta` trading-math library, native trainer, market-data frames |
| `media` | `yt-dlp>=2024.4.9` | Video/audio download from ~1800 sites (`nomorals/media`, `core/verify.py`) |
| `media-edit` | `Pillow>=10.0`, `opencv-python-headless>=4.8`, `scikit-image>=0.22` | Full image/video editing engine: Pillow ops (`images.py`), OpenCV frame-level video processing (`cv_video.py`), advanced cv ops — denoise, edge-detect, inpaint, seamless clone, cartoonize (`cv_ops.py`). ffmpeg binary still required from PATH for muxing/transcoding |
| `train` | `torch>=2.1`, `transformers>=4.38`, `datasets>=2.17`, `peft>=0.9` | GPU fine-tuning (`nomorals/training/finetune.py`, backends) |
| `finetune` | `unsloth`, `trl>=0.8` | Unsloth / QLoRA training backends (`nomorals/training/backends/unsloth.py`) |
| `hub` | `huggingface-hub>=0.22` | Resumable cached HF downloads (`voice/fetch.py`, media generation, `llm/download.py` fallback) |
| `ta` | `numpy>=1.24`, `pandas>=1.5`, `matplotlib>=3.7`, `scikit-learn>=1.3` | Trading-math core: indicators, regime, backtest (`nomorals/ta`) |
| `datasci` | `pandas>=1.5`, `matplotlib>=3.7` | Data workspace + plots (`nomorals/datasci`) |
| `docs` | `python-docx>=1.1`, `openpyxl>=3.1`, `pytesseract>=0.3`, `pdf2image>=1.17` | Office docs + OCR + PDF rendering (`nomorals/documents`) |
| `voice` | `edge-tts>=6.1`, `openai-whisper>=20231117`, `sounddevice>=0.4` | Cloud TTS, local Whisper STT, mic I/O |
| `chat` | `telethon>=1.36`, `discord.py>=2.3` | Telegram MTProto userbot, Discord bot (`nomorals/social/chat/`) |
| `cloud` | `boto3>=1.34`, `paramiko>=3.4` | AWS connector, SSH tooling |
| `genimg` | `diffusers>=0.27`, `rembg>=2.0` | Local image generation, background removal (`nomorals/media_edit/generate.py`) |
| `browser` | `playwright>=1.42` | Playwright browser automation (`nomorals/browser/service.py`) |
| `config` | `pyyaml>=6.0` | YAML configs (role specs, skill evolution, `core/decoder.py`) |
| `dev` | `pytest>=7.4`, `ruff>=0.3` | Test suite and linting |
| `all` | everything above | One command, everything works |

## Beyond extras — per-feature installs

These are real imports in the code, grouped by feature. Each group is one
command. (They're not in `pyproject.toml` extras because they're
heavyweight, GPU-bound, account-gated, or niche — install the groups you
actually use.)

### Chat / social

| Feature | Command | Used by |
|---|---|---|
| Telegram bot API + userbot | `pip install telethon` | `nomorals/social/chat/telegram.py` (Bot API over HTTP via stdlib `nomorals/core/http.py` — no `requests` needed; MTProto userbot via `telethon`), `nomorals/exporter.py` |
| Discord | `pip install discord.py` | `nomorals/social/chat/discord.py` |
| WhatsApp bridge | `cd bridge && npm install` (needs **Node ≥ 18**) | `bridge/whatsapp-bridge.mjs` — npm deps `@whiskeysockets/baileys`, `qrcode-terminal`; speaks JSON-lines to Devon's WhatsApp adapter over localhost TCP |

### Web search

| Feature | Command | Used by |
|---|---|---|
| DuckDuckGo keyless fallback | `pip install ddgs` | `nomorals/search/web.py` (`web_ddgs` source — multi-engine text search, no API key); missing → the source reports unavailable, other sources still work |

### Memory — vector store

| Feature | Command | Used by |
|---|---|---|
| Exact in-database KNN | `pip install sqlite-vec` | `nomorals/memory/vector_backends.py` (`SqliteVecBackend`) — fastest semantic recall; missing → `usearch`, then the zero-dependency legacy brute-force store |

### WisdomKeeper embeddings

| Feature | Command | Used by |
|---|---|---|
| ONNX embedding backend | `pip install fastembed` (pulls `onnxruntime`) | `nomorals/wisdom/embeddings.py` — local embeddings without torch |
| SentenceTransformers backend | `pip install sentence-transformers` | `nomorals/wisdom/embeddings.py` — needs `torch`; highest quality local embeddings |

### Media library

| Feature | Command | Used by |
|---|---|---|
| Rich audio tag reading | `pip install mutagen` | `nomorals/media/library.py` — MP3/FLAC/OGG/M4A tag metadata; missing → stdlib-only fallback |

### Browser automation

| Feature | Command | Used by |
|---|---|---|
| Playwright browser service | `pip install playwright && playwright install chromium` | `nomorals/browser/service.py` |

On a fresh Linux box the browser also needs OS libs:
`playwright install --with-deps chromium` (or `sudo apt install` the
missing libs it reports).

### Documents

| Feature | Command | Used by |
|---|---|---|
| OCR (Tesseract) + PDF rendering + Office docs | `pip install pytesseract pdf2image python-docx openpyxl Pillow` | `nomorals/documents/ocr.py` (needs the `tesseract` binary + `poppler-utils`), `nomorals/documents/parsers.py` |

### Data science & trading

| Feature | Command | Used by |
|---|---|---|
| Data workspace + plots | `pip install pandas matplotlib` | `nomorals/datasci/workspace.py`, `nomorals/datasci/plots.py` |
| Trading-math core | `pip install numpy pandas scikit-learn joblib` | `nomorals/ta/` (indicators, regime, backtest, meta), `nomorals/integrations/market_data.py` |
| Live crypto feeds | `pip install ccxt` | `nomorals/integrations/sentinel_bridge.py` hint |
| Live stock/forex feeds | `pip install yfinance` | `nomorals/integrations/sentinel_bridge.py` hint |
| YAML configs | `pip install PyYAML` | `nomorals/agents/role_specs.py`, `nomorals/agents/skill_evolution.py`, `nomorals/core/decoder.py` |

`scipy` is also named in the sentinel bridge's install hint for the full
stack: `pip install numpy pandas scipy scikit-learn pyyaml joblib`.

### Connectors & network tools

All 30+ connectors (`nomorals/connectors/`: GitHub, Gmail, Discord,
Telegram, Slack, Notion, Plaid, Mono, Paystack, Twilio, Spotify, YouTube,
Jumia/Konga/Jiji, …) are **stdlib-only** (urllib) — no pip packages, no
per-service SDKs. Credentials live in Devon's encrypted vault, not in
pip. Exceptions:

| Feature | Command | Used by |
|---|---|---|
| AWS | `pip install boto3` | `nomorals/connectors/aws.py` |
| SSH tooling | `pip install paramiko` | `nomorals/tools/attacker.py` |
| SOCKS proxies | `pip install PySocks` | `nomorals/tools/proxy.py` (`import socks`) |
| Builders proxy crypto | `pip install cryptography` | `nomorals/builders_proxy.py` |

### Voice — TTS engines (`nomorals/voice/tts.py`)

Pick the engines you want; each is independent.

| Engine | Command | Notes |
|---|---|---|
| XTTS v2 (local voice cloning) | `pip install TTS` | Coqui TTS; needs `torch` |
| Piper (CPU, MIT) | `pip install piper-tts` | + `python -m piper.download_voices en_US-lessac-medium` |
| Chatterbox (MIT) | `pip install chatterbox-tts` | Python 3.11+; expressive, incl. multilingual |
| Qwen3-TTS (0.6B) | `pip install qwen-tts` | |
| F5-TTS | `pip install f5-tts` | Needs a reference clip |
| Bark | `pip install git+https://github.com/suno-ai/bark.git` | Suno Bark |
| Kokoro | `pip install kokoro` | |
| CosyVoice | `pip install cosyvoice` | Model dir via `COSYVOICE_MODEL_DIR` |
| Orpheus | `pip install orpheus-speech` | GPU |
| OmniVoice | `pip install omnivoice` | + `torch`; GPU recommended, CPU offload supported |
| Dia | `pip install git+https://github.com/nari-labs/dia.git` | GPU-only, ~10 GB VRAM; model via `DIA_MODEL_ID` (default `nari-labs/Dia-1.6B-0626`) |
| Edge TTS (cloud) | `pip install edge-tts` | Also used by `nomorals/tools/audio.py` and `nomorals/integrations/voice_integration.py` |
| gTTS (cloud) | `pip install gTTS` | `nomorals/integrations/voice_integration.py` |
| HF serverless TTS | `pip install huggingface_hub` | Model/endpoint via `HF_TTS_MODEL` / `HF_TTS_ENDPOINT_URL` + `HF_TOKEN` |
| Voice-model fetch | `pip install huggingface_hub` | `nomorals/voice/fetch.py` |

### Voice — STT (`nomorals/integrations/stt.py`)

| Engine | Command | Notes |
|---|---|---|
| Faster-Whisper (local) | `pip install faster-whisper` | CTranslate2 — 7.7% WER, tiny decoder stack, good on weak hardware |
| Parakeet via onnx-asr (local) | `pip install onnx-asr` (optionally `onnx-asr[cpu,hub]`) | NVIDIA Parakeet TDT 0.6B — 6.3% WER English, fastest free dictation |
| whisper.cpp (local) | `pip install pywhispercpp` | No torch; GGUF/quantized models |
| Whisper (local) | `pip install openai-whisper` | `import whisper` — classic reference, slowest, kept as fallback |
| OpenAI Whisper API | `pip install openai` | Key via `NM_AUDIO_STT_API_KEY` / `OPENAI_API_KEY` |
| SpeechRecognition | `pip install SpeechRecognition` | |
| AssemblyAI (cloud) | `pip install assemblyai` | |

### Voice — microphone I/O

| Feature | Command | Used by |
|---|---|---|
| Mic capture/playback | `pip install sounddevice` | `nomorals/voice/session.py` — needs the PortAudio system library (see below) |

Offline system-TTS fallbacks need no pip packages: `espeak-ng`,
`pico2wave`, or macOS `say` (`nomorals/tools/audio.py` probes them from
PATH).

### Media generation (`nomorals/media_edit/generate.py`)

| Backend | Command | Select with |
|---|---|---|
| Local (Stable Diffusion etc.) | `pip install diffusers torch` | `MEDIA_GEN_BACKEND=diffusers` |
| HF serverless inference | `pip install huggingface_hub` + `HF_TOKEN` | `MEDIA_GEN_BACKEND=hf` (default `auto`) |
| Background removal | `pip install rembg` | — |

### Training — notebook/Colab flow extras

`nomorals/training/finetune.py` prints these hints for the Kaggle/Colab
path (beyond the `train`/`finetune` extras):

```bash
pip install -q transformers peft trl accelerate bitsandbytes datasets
pip install -q unsloth bitsandbytes datasets
pip install -q pyarrow gguf
```

LlamaFactory backend hint (`nomorals/training/backends/llama_factory.py`):
`pip install llamafactory[torch]` — or use the unsloth backend on Colab.

### Sentinel trading-bot bridge

`nomorals/integrations/sentinel_bridge.py` and `nomorals/tools/trading.py`
optionally import the user's own `sentinel` package (their Sentinel.py
trading bot — `from sentinel.core.engine import QuantumEngine`). It is
**not on PyPI**; install/provide it from the user's own source. Everything
else in the trading stack degrades gracefully without it.

## System dependencies (binaries — not pip)

These are OS-level binaries Devon shells out to or probes via
`shutil.which`. Install with your system package manager; pip can't
provide them.

| Binary | Used by | Ubuntu/Debian | macOS (brew) |
|---|---|---|---|
| `ffmpeg` + `ffprobe` | `nomorals/media_edit/videos.py` (muxing/transcoding), `nomorals/media/playback.py` | `sudo apt install ffmpeg` | `brew install ffmpeg` |
| `tesseract` | `nomorals/documents/ocr.py` (+ `llm/providers/ocr.py` probe); override path with `NM_OCR_BINARY` | `sudo apt install tesseract-ocr` | `brew install tesseract` |
| `pdftoppm` (poppler) | `pdf2image` PDF page rendering (`documents/ocr.py`) | `sudo apt install poppler-utils` | `brew install poppler` |
| `git` | `nomorals/codews/workspace.py`, backup versioning/push | `sudo apt install git` | `brew install git` |
| `yt-dlp` (CLI) | `nomorals/core/verify.py` fallback when the module is absent; `nomorals/media/playback.py` YouTube search fallback when the module is absent | `pip install yt-dlp` or `sudo apt install yt-dlp` | `brew install yt-dlp` |
| `yt-dlp` (Python module) | `nomorals/media/playback.py` YouTube search + audio extraction (`play_youtube`, `/play youtube:`, `nm music play --youtube`); missing → honest error with install hint | `pip install yt-dlp` | `pip install yt-dlp` |
| `fluidsynth` (CLI) | `nomorals/media/synth_backend.py` — studio-quality composed-song rendering on pc/vps/workstation profiles (with a soundfont); missing → pure-Python builtin synth | `sudo apt install fluidsynth` (soundfont: `nm music soundfont install` fetches GeneralUser GS, ~31 MB, free) | `brew install fluidsynth` |
| `llama-server` | `nomorals/llm/local_server.py` — local GGUF inference | [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) (prebuilt binary) | same |
| `bwrap` / `unshare` | `nomorals/execbox.py` sandbox isolation (bubblewrap preferred, namespaces fallback) | `sudo apt install bubblewrap` (`unshare` ships with `util-linux`) | not applicable — Linux-only |
| `espeak-ng` / `pico2wave` | `nomorals/tools/audio.py` offline TTS fallback | `sudo apt install espeak-ng` / `libttspico-utils` | `say` is built in |
| PortAudio (`libportaudio2`) | `sounddevice` mic I/O (`voice/session.py`) | `sudo apt install libportaudio2` (`portaudio19-dev` if building) | `brew install portaudio` |
| `node` ≥ 18 + `npm` | `bridge/whatsapp-bridge.mjs` (WhatsApp) | [nodejs.org](https://nodejs.org) or `sudo apt install nodejs npm` | `brew install node` |
| `whisper.cpp` binary (optional) | Local STT alternative; paths via `NM_WHISPER_CPP_BIN` / `NM_WHISPER_CPP_MODEL` | [whisper.cpp releases](https://github.com/ggerganov/whisper.cpp/releases) | same |

## API keys / env vars

Names only — never values. Set the ones for the features you use.

| Area | Variables |
|---|---|
| LLM providers | `NM_OPENAI_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN` (also accepted as `HUGGING_FACE_HUB_TOKEN` / `NM_HF_TOKEN`), `NM_HF_BASE`, `NM_HF_SERVER` (llama.cpp provider base URL) |
| Voice / TTS | `HF_TTS_MODEL`, `HF_TTS_ENDPOINT_URL`, `DIA_MODEL_ID`, `ORPHEUS_MODEL_ID`, `COSYVOICE_MODEL_DIR`, `NOMORALS_VOICES_DIR` |
| Voice / STT | `NM_AUDIO_STT_API_KEY`, `NM_AUDIO_STT_BASE_URL`, `NM_WHISPER_CPP_BIN`, `NM_WHISPER_CPP_MODEL` |
| Telegram | `NM_TELEGRAM_BOT_TOKEN` (or `TELEGRAM_BOT_TOKEN`), `TELEGRAM_API_BASE`; the telethon userbot path takes `api_id`/`api_hash` (from [my.telegram.org](https://my.telegram.org)) |
| Connectors | Credentials live in Devon's encrypted vault, unlocked with `NM_VAULT_KEY` / `NM_VAULT_PASSPHRASE` — not in env. A few connectors also honor env: `TELEGRAM_BOT_TOKEN`, `DISCORD_BOT_TOKEN`, `GITHUB_TOKEN` / `NM_API_GITHUB_TOKEN`, `PLAID_ACCESS_TOKEN`, `PLAID_ENV` |
| Market / data | `FOOTBALL_DATA_API_KEY`, `ODDS_API_KEY`, `DEVON_WEATHER_PLACE` |
| CAPTCHA (human-in-the-loop) | `CAPTCHA_API_KEY`, `CAPTCHA_API_URL`, `NM_CAPTCHA_SOLVER` |
| WhatsApp bridge | `NM_WHATSAPP_BRIDGE_HOST`, `NM_WHATSAPP_BRIDGE_PORT` |
| Media gen | `MEDIA_GEN_BACKEND` (`auto`/`hf`/`diffusers`/`off`), `MEDIA_GEN_MODEL`, `MEDIA_GEN_DIFFUSERS_MODEL`, `MEDIA_GEN_PROVIDER`, `NM_MEDIA_VIDEO_PRESET`, `NM_MEDIA_AUDIO_BITRATE` |
| OCR | `NM_OCR_BINARY` (custom tesseract path) |
| Proxies | `NM_PROXY_URL` |

## What's NOT a Devon dependency

Verified during the scan — don't install these for Devon itself:

- **`flask` / `fastapi` / `django` / `telegram` / `react` imports** found in
  `nomorals/builders/` are *generated-project code* (inside template
  strings in `app_builder.py` and `builders/templates/`), not Devon
  runtime imports. The shipped templates are stdlib-only
  (see each template's `requirements.txt`); a generated project that uses
  a web framework needs those packages installed **in the generated
  project**, not in Devon.
- **All 30+ connectors are stdlib-only** (urllib) except `boto3`
  (AWS). No per-service SDKs to install.
- `nm doctor` (via `nomorals/compat.py`) is the single place that probes
  optional features; every consumer has a pure-stdlib fallback, so a
  missing package degrades the feature — it never breaks the core.

## Verifying your setup

```bash
nm doctor              # which optional deps + binaries are present
python -c "import nomorals"   # core imports with zero third-party packages
pytest tests/ -x -q    # full suite (needs .[dev] plus the extras you use)
```
