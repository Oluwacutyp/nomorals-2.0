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

From PowerShell on the RDP.

**FIRST — if `C:\devon\nomorals-2.0` already exists, SKIP clone.**
Cloning into an existing directory fails with
`fatal: destination path already exists and is not an empty directory`.
Do this instead:

```powershell
# Already have a checkout? SKIP clone — cd in and pull:
cd C:\devon\nomorals-2.0
git pull origin main
```

**Only if the directory does NOT exist yet** (first time ever):

```powershell
git clone https://github.com/Oluwacutyp/nomorals-2.0.git C:\devon\nomorals-2.0
cd C:\devon\nomorals-2.0
```

Then, **from `C:\devon\nomorals-2.0`** (not from `C:\devon` — running
`.\scripts\...` from the parent directory gives "not recognized"):

```powershell
# 1. Preview (changes nothing):
.\scripts\deploy_devon_ec2_rdp.ps1 -DryRun

# 2. Deploy (pulls latest main, runs the real installer, writes .env from template):
.\scripts\deploy_devon_ec2_rdp.ps1

#    Variants:
#    .\scripts\deploy_devon_ec2_rdp.ps1 -IncludeLocalWhisper   # also Whisper-tiny STT fallback
#    .\scripts\deploy_devon_ec2_rdp.ps1 -SkipInstall            # installer already ran
#    .\scripts\deploy_devon_ec2_rdp.ps1 -RepoPath D:\devon\nomorals-2.0
```

## Recovery — if you already hit the failed-clone / wrong-directory error

Symptoms: `fatal: destination ... is not an empty`,
`.\scripts\deploy_devon_ec2_rdp.ps1` "not recognized",
`notepad .env` opened an **empty** `.env`, `.\.venv\Scripts\python.exe`
"not recognized". That chain means: the clone failed (dir already
existed), so the deploy script never ran, so no `.env` was copied and
no venv was created — and the later commands ran from `C:\devon`
instead of `C:\devon\nomorals-2.0`. Fix with these exact commands:

```powershell
# 1. Get into the repo (the checkout already exists — do NOT re-clone):
cd C:\devon\nomorals-2.0
git pull origin main

# 2. If the failed run left an EMPTY .env in the WRONG directory, remove it
#    (only if it is 0 bytes — check first, never delete a filled .env):
#    Get-Item C:\devon\.env   # confirm 0 bytes
#    Remove-Item C:\devon\.env

# 3. Run the deploy from the repo directory. It pulls latest main, runs the
#    installer, and writes .env from the template (an empty .env in the repo
#    is backed up to .env.empty.bak and replaced — never left as "done"):
.\scripts\deploy_devon_ec2_rdp.ps1

# 4. Fill .env (from the repo directory, so notepad opens the REAL file):
notepad .env

# 5. Start Devon (from the repo directory):
.\.venv\Scripts\python.exe -m nomorals
```

## Recovery — working-tree damage: `scripts/` missing / many D (deleted) files

Symptoms: `git status` shows a long list of `D` (deleted) files —
`tests/test_voice_*.py`, `tests/test_wave*.py`, `training/*`,
`vendor/sentinel`, `workspace/*`, etc. — and
`dir scripts\deploy*` fails with
`Cannot find path 'C:\devon\nomorals-2.0\scripts' because it does not exist.`
The whole `scripts/` folder is gone from the working tree. `git pull`
says "Already up to date" at the same time — because **pull only moves
HEAD, it does NOT restore deleted working-tree files.** Pull + missing
script coexisting is the signature of this damage.

Fix with these exact commands:

```powershell
# 1. Restore ALL deleted tracked files (this is the fix pull cannot do):
cd C:\devon\nomorals-2.0
git restore .

# 2. THEN pull latest main:
git pull origin main

# 3. Confirm the damage is gone before running anything else:
git status --short   # should show no long D (deleted) list
dir scripts\deploy*  # should list deploy_devon_ec2_rdp.ps1
```

Rules:

- Use `git restore .` for tracked deletions. It restores deleted
  tracked files and leaves untracked files alone.
- Do **NOT** run `git clean -fd` blindly — it deletes untracked files,
  including `.env`. Never clean before checking what is untracked
  (`git status` first, and only clean paths you have confirmed are
  safe to delete).
- If `git restore .` still leaves `scripts/` missing, the clone itself
  is partial — a fresh clone into an **EMPTY** directory is the only
  fix. Back up any filled `.env` first, then clone into an empty dir
  and copy the backed-up `.env` back.

### Fresh-clone fallback — when `git restore .` fails or pull aborts

If `git restore .` fails with
`error: pathspec '.' did not match any file(s) known to git`, or
`git pull` aborts with
`error: Your local changes to the following files would be overwritten by merge:`
for files that do **not** exist on disk (they are DELETED in the working
tree, which git counts as a "local change", so the incoming commit that
touches those same files cannot merge) — the checkout is in a
contradictory state (possibly sparse checkout, corrupt index, or a
partially damaged `.git`). Do not fight it. Fresh-clone beside it:

```powershell
# 1. Rename the broken checkout aside — do NOT delete it until the
#    fresh clone is verified working (it may hold a filled .env):
cd C:\devon
Rename-Item C:\devon\nomorals-2.0 C:\devon\nomorals-2.0-broken

# 2. Fresh clone into an EMPTY directory:
git clone https://github.com/Oluwacutyp/nomorals-2.0.git C:\devon\nomorals-2.0-fresh
cd C:\devon\nomorals-2.0-fresh

# 3. Confirm the damage is gone before running anything else:
dir scripts\deploy*  # must list deploy_devon_ec2_rdp.ps1

# 4. Deploy from the fresh checkout:
.\scripts\deploy_devon_ec2_rdp.ps1
```

Rules:

- Do **NOT** run `git clean -fd` on the broken checkout — it deletes
  untracked files, including a filled `.env`.
- Do **NOT** delete `nomorals-2.0-broken` until the fresh clone is
  verified working AND any filled `.env` has been copied over (back up
  the filled `.env` first, then copy it into the fresh checkout —
  never paste secrets in chat).
- Once the fresh clone is verified working, the broken directory can
  be deleted.

If `git pull` fails with divergent history, the script prints the exact
recovery and stops — it does **not** auto-reset (reset discards local
changes). Follow the printed recovery, then re-run the deploy.

What the wrapper does:

1. Pull (`git pull --ff-only`) if the repo exists, clone only if it
   does not. A non-empty directory without `.git` is refused with a
   clear message instead of a bare git error. A failed pull prints the
   exact recovery and stops — never an auto-reset.
2. Call `scripts/install_devon_windows.ps1` — the real installer. The
   wrapper never duplicates it.
3. Copy `scripts/devon_ec2_env.template` → `.env` when `.env` is
   missing. A non-empty existing `.env` is never overwritten. An
   **empty** (0-byte) `.env` is backed up to `.env.empty.bak` and
   replaced from the template, then verified (non-empty, placeholder
   keys present) — never left as an empty "done".
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
| `fatal: destination path already exists and is not an empty directory` | Directory already exists — **SKIP clone**. See "Recovery" above: `cd C:\devon\nomorals-2.0` + `git pull origin main`. |
| `.\scripts\deploy_devon_ec2_rdp.ps1` "not recognized" | Wrong directory — you are in `C:\devon`, not `C:\devon\nomorals-2.0`. Run `cd C:\devon\nomorals-2.0` first, then re-run. |
| `notepad .env` opened an empty `.env` | It opened `.env` in the wrong directory (or an empty one). Close it, `cd C:\devon\nomorals-2.0`, re-run deploy — an empty repo `.env` is backed up to `.env.empty.bak` and replaced from the template. Delete a stray empty `C:\devon\.env` only after confirming 0 bytes. |
| `.\.venv\Scripts\python.exe` "not recognized" | No venv — the installer never ran (deploy script never ran). Run the deploy from `C:\devon\nomorals-2.0` first. |
| `git pull --ff-only` fails (divergent) | Script prints exact recovery and stops — it does not auto-reset. Follow the printed recovery, then re-run. |
| `git status` shows many `D` (deleted) files / `scripts/` missing, but `git pull` says "Already up to date" | Working-tree damage — pull does NOT restore deleted files. Run `git restore .`, then `git pull origin main`. Never `git clean -fd` blindly (deletes untracked files incl. `.env`). See "Recovery — working-tree damage". |
| `git restore .` fails (`pathspec '.' did not match`) OR pull aborts with "local changes would be overwritten" for files that don't exist on disk | Contradictory checkout — do not fight it. Fresh-clone fallback: rename broken dir to `nomorals-2.0-broken` (do NOT delete), clone into `nomorals-2.0-fresh`, verify `dir scripts\deploy*`, deploy from the fresh checkout. See "Fresh-clone fallback". |
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
