# Devon on Windows RDP — Install Guide

## How well will it work on the current RDP?

Honest answer, by tier. The current GoldenSwing EC2 host is a small
CPU-only Windows box (no GPU, a few vCPUs, a few GB RAM). That maps to
**LOW** in the installer.

| What | LOW (current RDP) | MEDIUM (8 GB) | HIGH (16 GB+ / GPU) |
|---|---|---|---|
| Chat, tools, games, books, learn, legal, commitments, decisions, focus, family, immune, reviews, preferences | ✅ Works well | ✅ | ✅ |
| Telegram / Discord / WhatsApp adapters | ✅ Once `.env` keys set | ✅ | ✅ |
| Local LLM (GGUF / llama.cpp / Ollama) | ❌ Not installed — owner decision: API keys only | ❌ | ❌ |
| LLM via API (Groq / OpenRouter / OpenAI / HF) | ✅ Primary brain on all tiers | ✅ | ✅ |
| Speech-to-text / TTS | ✅ API first; Piper local fallback | ✅ | ✅ |
| Voice cloning (XTTS) | ⏭️ Skipped — needs better hosting, API fallback | ⏭️ | ⏭️ |
| Image diffusion | ⏭️ API / deferred — local diffusion needs better hosting | ⏭️ | ⏭️ |
| Video generation | ⏭️ Deferred — needs better hosting | ⏭️ | ⏭️ |
| Trading | Paper only, always. Real trading lives in GoldenSwing, never Devon | Same | Same |

**Core Devon works well on the current RDP.** Heavy media is what
suffers. The installer is profile-aware: it installs the small models
on LOW and tells you exactly what is degraded, instead of failing or
silently installing 17 GB of models that OOM.

**GoldenSwing + Devon on the same host:** watch RAM. Run one heavy
job at a time. Do not run a GoldenSwing backtest and a Devon diffusion
render together on a small host. The live MT5 runner is light; a
backtest is not.

## Install

From the repo root, in PowerShell:

```powershell
.\scripts\install_devon_windows.ps1
```

Options:

```powershell
.\scripts\install_devon_windows.ps1 -DryRun                  # preview, change nothing
.\scripts\install_devon_windows.ps1 -IncludeLocalWhisper    # also local Whisper-tiny fallback
.\scripts\install_devon_windows.ps1 -SkipLocalExtras        # API-only, no local extras
.\scripts\install_devon_windows.ps1 -Force                   # regenerate .env (backed up)
```

What the script does:

1. **Preflight** — OS / CPU / RAM / disk free / GPU / Python →
   LOW (≤4.5 GB RAM or <2 vCPU) / MEDIUM (8 GB) / HIGH (16 GB+ or GPU).
   Tier selects models and pip extras. It never gates capability.
2. **System deps via winget** — Python 3.11, Git, ffmpeg, espeak-ng,
   Visual C++ Redistributable, nano. No admin required where possible.
3. **Python venv + pip extras by tier** — base always; media-edit /
   voice / TA / genimg on MEDIUM+; train / cloud / dev on HIGH.
   Core Devon has zero mandatory third-party deps (`pyproject.toml`:
   `dependencies = []`); everything is optional acceleration.
4. **API-first, no local LLM.** LLM / STT / TTS / diffusion all go
   through API keys in `.env`. **No GGUF, no llama.cpp, no Ollama** —
   owner decision, till better hosting is affordable. The only local
   downloads are tiny fallbacks: Piper TTS (~0.1 GB, always) and
   Whisper-tiny (~0.1 GB, only with `-IncludeLocalWhisper`). XTTS /
   diffusion / RVC are skipped on RDP — API fallback / deferred.
   Failed downloads warn and continue; Devon runs on API fallback.
5. **`.env` + `config/*.yaml`** — `.env` from `.env.example` plus
   grouped keys, placeholders only, never real secrets. Existing
   `.env` is kept unless `-Force` (then backed up first).
6. **Verification** — `python -m nomorals --help`, `nm doctor`, then a
   capability matrix: what works / what is degraded on this host.

## Models — API-first (owner decision)

**No GGUF. No local LLM. No llama.cpp. No Ollama.** LLM / STT / TTS /
diffusion all use API keys in `.env`, till better hosting is affordable.
Profile = API-first, online-first; offline is fallback, not primary.

| Local extra | What | Size | Default |
|---|---|---|---|
| Piper TTS (`rhasspy/piper-voices` en_US-lessac-medium) | Local TTS fallback so voice works with no keys | ~0.1 GB | Always |
| Whisper-tiny (`openai/whisper-tiny`) | Local STT fallback | ~0.1 GB | Only with `-IncludeLocalWhisper` |

| Deferred (needs better hosting) | For now |
|---|---|
| XTTS v2 voice cloning (private personal use) | API fallback |
| Image / video diffusion (SD-Turbo, SDXL-Turbo) | API / deferred |
| RVC / Demucs / stems | Deferred |

Local extras need <1 GB free. XTTS v2 is for private personal use
(non-commercial OK); Chatterbox (MIT) is the public-facing voice.

## Config — `nano` / `notepad`

On Windows, `nano` is not native. The installer installs it via
winget (`GNU.Nano`). If you prefer the built-in editor:

```powershell
notepad .env     # Windows built-in
nano .env        # if nano installed (same as Termux)
```

On Termux: `nano .env` as usual.

`.env` is gitignored. Never put real secrets in chat, screenshots,
or git. Devon boots fully offline against the Mock provider with no
keys at all — fill in only the keys you use:

- `NM_CHAT_TELEGRAM_BOT_TOKEN` — gated bot
- `NM_CHAT_TELEGRAM_PERSONAL_BOT_TOKEN` — personal bot
- `NM_GROQ_API_KEY` / `NM_OPENROUTER_API_KEY` — cloud brains
  (primary on all tiers; there is no local LLM)
- `HF_TOKEN` — Hugging Face downloads (public models work without it)

Per-module templates live in `config/`: `models.yaml`, `chat.yaml`,
`trading.yaml` (paper only, `allow_live: false` — never set true
from Devon).

## Start Devon

```powershell
.\.venv\Scripts\python.exe -m nomorals --help
.\.venv\Scripts\python.exe -m nomorals.cli doctor
.\.venv\Scripts\python.exe -m nomorals
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `winget not found` | Install App Installer from the Microsoft Store, re-run. |
| Python still old after install | Close + reopen PowerShell (PATH refresh), or run the `.venv` python directly. |
| Local extra download fails | Re-run the script; `hf download` resumes. Devon runs on API fallback without it. |
| `.env` lost / wrong | Re-run with `-Force`; the old one is backed up as `.env.bak-<timestamp>`. |
| Out of memory on LOW | Use `-Models minimal`, use Groq/OpenRouter as primary brain, never run backtest + diffusion together. |
| `nm doctor` fails | Devon still runs; read the `nm doctor` output — it names the missing piece. |
| Telegram bot not answering | Check `.env` tokens, `NM_CHAT_TELEGRAM_BOT_ENABLED=1`, and that only one process owns `getUpdates` (MT5 runner or Devon, never both). |

## What the script does NOT do

- No real secrets, ever. Placeholders only.
- No live trading. Devon is paper-only; real trading lives in GoldenSwing.
- No admin required where possible; winget may prompt for elevation
  for per-machine installs.
- No silent downgrades. Degraded paths are printed in the capability matrix.
