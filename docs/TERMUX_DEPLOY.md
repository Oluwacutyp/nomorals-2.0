# Devon on Termux — your phone becomes the server

Run the whole stack on your Samsung until Serv00 approves: the uncensored
8B GGUF served locally by llama.cpp (fully offline brain) + the Devon
Telegram bot talking to it. No cloud, no credits, no censorship.

## 0. What you need

- **Termux from F-Droid** (the Play Store version is abandoned — do not use it).
  Optional companion: **Termux:API** from F-Droid (only needed for extra
  phone integrations later).
- **WiFi** for the one-time ~4.9GB model download. Do this on WiFi, not MTN.
- About **8GB free storage** (4.9GB model + repo + slack).
- A **Telegram bot token** from [@BotFather](https://t.me/BotFather) and your
  numeric Telegram chat id (message [@userinfobot](https://t.me/userinfobot)).

## 1. Base packages

```bash
pkg update && pkg upgrade -y
pkg install -y python git curl wget
```

## 2. llama.cpp (prebuilt — no compiling)

Grab the latest Android arm64 build from
[llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
(the file looks like `llama-bXXXX-bin-android-arm64-v8a.zip`):

```bash
mkdir -p ~/llama.cpp && cd ~/llama.cpp
# replace the URL with the latest release asset:
curl -fSL -o llama-android.zip \
  https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-bXXXX-bin-android-arm64-v8a.zip
unzip -o llama-android.zip
chmod +x llama-server llama-cli
./llama-cli --version   # sanity check
```

## 3. The model (WiFi only)

```bash
mkdir -p ~/models && cd ~/models
# ~4.9GB — WiFi, not mobile data. The repo is private, so pass your HF token:
curl -fSL -C - -o dolphin-8b-merged.Q4_K_M.gguf \
  -H "Authorization: Bearer <your-hf-token>" \
  https://huggingface.co/Cutyp/dolphin-8b-merged/resolve/main/dolphin-8b-merged.Q4_K_M.gguf
```

> **Reality check:** an 8B model at full chat load makes phones hot and can
> stall — if yours throttles or goes unresponsive, that's the chip, not the
> setup. The 8B GGUF is still worth having (it runs great on a PC or a VPS),
> but the realistic everyday phone brain is the **3.8B** (`Cutyp/codebeast-3.8b`
> Q4_K_M, ~2.3GB) once training finishes — same steps, smaller file, far
> less heat. Until either is on the phone, the cloud API fallbacks in step 5
> keep Devon talking.

## 4. Devon herself

```bash
git clone https://github.com/Oluwacutyp/nomorals-2.0.git ~/devon
cd ~/devon && pip install -e .
# optional extras (skip torch — painful on Termux, not needed here):
pip install numpy yt-dlp Pillow
# for the userbot (your own Telegram account):
pip install telethon
chmod +x scripts/termux/start_termux.sh
```

## 5. Secrets file

Create `~/.devon-bot.env` (leading dot, `chmod 600`) with your real values:

```bash
# --- brain: local uncensored 8B via llama-server (offline, no credits) ---
NM_LLM_PROVIDER=llama_cpp
# llama_cpp_url defaults to http://localhost:8080 — no need to set it
# unless you changed llama-server's port.

# --- brain fallbacks when you have internet (used only if local dies) ---
NM_LLM_FALLBACK_CHAIN=hf_serverless,openrouter
NM_HF_MODEL=Qwen/Qwen2.5-7B-Instruct
HF_TOKEN=<your-hf-token>
NM_OPENROUTER_API_KEY=<redacted>
NM_OPENROUTER_MODEL=qwen/qwen3-8b:free

# --- chat: Telegram bot (long-polling, no webhook needed) ---
NM_PARTNER_PLATFORMS=telegram-bot
NM_CHAT_TELEGRAM_BOT_ENABLED=true
NM_CHAT_TELEGRAM_BOT_TOKEN=<paste-BotFather-token>
NM_CHAT_TELEGRAM_BOT_CHATS=<your-numeric-chat-id>

# --- chat: your own Telegram account (userbot, optional but recommended) ---
# Lets Devon read and reply as YOU, not as a bot — DMs, groups, channels.
# Needs: api_id + api_hash from https://my.telegram.org → API development,
# and: pip install telethon
# First boot is interactive: enter your number (e.g. 234803...), then the
# code Telegram sends you, then 2FA if you have it. After that the session
# file is the credential — chmod 600, never share it, never commit it.
NM_PARTNER_PLATFORMS=telegram-bot,telegram
NM_CHAT_TELEGRAM_ENABLED=true
NM_CHAT_TELEGRAM_API_ID=<your-api-id>
NM_CHAT_TELEGRAM_API_HASH=<your-api-hash>
NM_CHAT_TELEGRAM_SESSION=data/telegram.session
NM_CHAT_TELEGRAM_CHATS=<your-numeric-chat-id>

# --- proactive delivery (morning briefing, price alerts) ---
NM_PARTNER_PROACTIVE_ENABLED=true
```

Notes:
- There is deliberately **no uncensored cloud model** in the fallback chain:
  HF's and OpenRouter's free tiers serve curated catalogs only — no
  abliterated/uncensored models are offered on either (verified 2026-10-01).
  Uncensored stays local (the GGUF) until Serv00.
- `NM_HF_MODEL` is Qwen2.5-7B-Instruct — a working model on HF's
  inference API. Note: HF retired the old api-inference endpoint; only
  models in their hosted catalog work (microsoft/Phi-3.5-mini-instruct
  is NOT hosted there anymore).

## 6. Run it

```bash
~/devon/scripts/termux/start_termux.sh
```

This starts `llama-server` (the 8B brain, ~30s to load) and then the bot.
Logs go to `~/devon-termux.log`. Run `termux-wake-lock` first so Android
doesn't sleep the session.

## 7. Keep Android from killing it

1. Android Settings → Apps → Termux → Battery → **Unrestricted**
   (a.k.a. "Don't optimize").
2. In Termux: `termux-wake-lock` (acquire), and re-run the start script
   after reboots. For boot persistence later, Termux:Boot can run the
   script automatically.

## 8. Talking to her

- Telegram: message your bot. She answers with the local 8B.
- Sanity check the brain directly:
  `curl http://localhost:8080/health` → `{"status":"ok"}`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `llama-server: not found` | Redo step 2; make sure the zip's binaries are executable |
| Bot answers but slowly | 8B Q4 on phone CPU ≈ a few tok/s — normal; shorten replies via prompt |
| Bot exits when screen off | Step 7 — battery unrestricted + wake lock |
| `HF 401` in logs | Fallback only triggers with internet; check `HF_TOKEN` |
| Model file 404 | The repo is PRIVATE — the curl needs `-H "Authorization: Bearer <token>"` (in the doc). Filename may also differ — check the file list at huggingface.co/Cutyp/dolphin-8b-merged while logged in |
