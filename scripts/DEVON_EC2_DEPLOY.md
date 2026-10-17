# Devon on EC2 RDP — Deploy Guide (Devon ONLY)

Devon (nomorals-2.0) on a Windows EC2 RDP host. Devon only: Devon's repo,
Devon's `.env`, Devon's two bots. Nothing else on the host is installed,
configured, started, or touched by this deploy.

## What runs where

| Thing | Repo | Bots / config |
|---|---|---|
| Devon (this guide) | `nomorals-2.0` | Devon main bot (gated) + second Devon bot (user-only) |
| Any other system on the host | Its own repo | Its own config — NOT part of Devon, NOT in Devon's `.env` |

## Prerequisites (on the RDP)

- Windows Server / Windows 10+ with PowerShell 5.1+ (or PowerShell 7)
- Internet access (API-first: Devon calls LLM APIs)
- A secure store for your secrets (password manager / Secure Vault).
  **Never paste tokens/keys in chat, screenshots, or git.**

## Deploy — exact commands

From PowerShell on the RDP:

```powershell
# 1. Get the deploy wrapper (first time only). If you already have a checkout, skip to step 2.
git clone https://github.com/Oluwacutyp/nomorals-2.0.git C:\devon\nomorals-2.0
cd C:\devon\nomorals-2.0

# 2. Preview (changes nothing):
.\scripts\deploy_devon_ec2_rdp.ps1 -DryRun

# 3. Deploy (pulls latest main, runs the real installer, writes .env from template):
.\scripts\deploy_devon_ec2_rdp.ps1

#    Variants:
#    .\scripts\deploy_devon_ec2_rdp.ps1 -IncludeLocalWhisper   # also Whisper-tiny STT fallback
#    .\scripts\deploy_devon_ec2_rdp.ps1 -SkipInstall            # installer already ran
#    .\scripts\deploy_devon_ec2_rdp.ps1 -RepoPath D:\devon\nomorals-2.0
```

What the wrapper does:

1. Clone (or `git pull --ff-only`) the `nomorals-2.0` repo, `main`.
2. Call `scripts/install_devon_windows.ps1` — the real installer. The
   wrapper never duplicates it.
3. Copy `scripts/devon_ec2_env.template` → `.env` **only when `.env`
   is missing**. An existing `.env` is never overwritten.
4. Verify: `python -m nomorals --help`, `nm doctor`,
   `nm impossible examples` (Impossible Made Possible CLI smoke).
5. Print the start commands.

## Fill `.env` — secure entry only

```powershell
notepad .env
```

Fill **only** the keys you use, from your secure store. Required on EC2:

| Key | What |
|---|---|
| `NM_GROQ_API_KEY` | Primary brain (Groq, free, no card). Recommended. |
| `NM_CHAT_TELEGRAM_BOT_TOKEN` | **Devon main bot — GATED (stays gated).** From @BotFather. |
| `NM_CHAT_TELEGRAM_PERSONAL_BOT_TOKEN` | **Second Devon bot — USER-ONLY (owner only).** From @BotFather, a *different* bot from main. |

Rules:

- Two Devon bots = two **separate** tokens, two separate `getUpdates`
  pollers. Never share one token across two bots (two pollers on one
  token = missed commands / 409).
- Main bot: `NM_CHAT_TELEGRAM_BOT_PARTNER_CONTEXT=0` (gated chats get no
  owner-private relationship context).
- Second bot: `NM_CHAT_TELEGRAM_PERSONAL_BOT_PARTNER_CONTEXT=1`
  (owner-only, full context).
- The full key list lives in `scripts/devon_ec2_env.template` — every
  config key Devon code reads, grouped, placeholders only, never real
  values. Optional keys stay blank/commented when unused.
- `.env` is gitignored. Never commit it.

## Start Devon

```powershell
cd C:\devon\nomorals-2.0
.\.venv\Scripts\python.exe -m nomorals
```

Check both bots answer:

- Main bot (gated) answers in its gated chats.
- Second bot (user-only) answers only the owner.

## Verify — checklist

| Check | Expected |
|---|---|
| `python -m nomorals --help` | Prints help, exit 0 |
| `python -m nomorals.cli doctor` | Lists adapters; both Telegram bots `enabled` once tokens are set |
| `nm impossible examples` | Car + software first-principles worked examples |
| `/impossible solve <problem>` (in chat) | Full dossier |
| `/impossible sources` | 25-source digestion registry |
| `/impossible cross <problem>` | Impossible + WisdomKeeper + BookForge cross-query |
| Singing | Devon returns an **audio FILE** (see below) |

## Singing on EC2

Yes — singing works on EC2. The voice/music stack (`nomorals/voice/`,
DiffSinger singing backend) generates singing to an **audio file**.
EC2 has no speaker, so there is no live-speaker playback — the output
is a file you send via Telegram or save to disk. Same engine, same
stack, file output. Verify by asking Devon to sing and confirming an
audio file comes back.

Local voice on EC2 is API-first: LLM / STT / TTS / diffusion via API
keys. The only local downloads are tiny fallbacks (Piper TTS always,
Whisper-tiny only with `-IncludeLocalWhisper`). **No GGUF, no local
LLM, no llama.cpp, no Ollama** — owner decision, till better hosting
is affordable. XTTS / diffusion / RVC are skipped on RDP (API fallback).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `winget not found` | Install App Installer from the Microsoft Store, re-run. |
| Python still old after install | Close + reopen PowerShell (PATH refresh). |
| `.env` lost / wrong | Re-run deploy; existing `.env` is kept. To regenerate, back it up first, delete it, re-run. |
| Bot not answering | Check its token in `.env`, `NM_CHAT_TELEGRAM_*_ENABLED=1`, and that only **one** process owns that token's `getUpdates`. |
| 409 / missed commands | Two pollers on one token. Give each bot its own token. |
| Out of memory | One heavy job at a time on a small host. Devon is light; media renders are not. |
| `nm doctor` fails | Devon still runs; read the output — it names the missing piece. |

## What this deploy does NOT do

- No real secrets, ever. Placeholders only.
- No other product's install, config, start, or data. Devon only.
- No silent downgrades. Degraded paths print in the capability matrix.
