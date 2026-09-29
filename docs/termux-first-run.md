# NoMorals Core — First Trail Run on Termux

A copy-paste path from a fresh Android phone to your companion talking on the
local console, fully offline, and (optionally) on Telegram. Every command is
exactly what you type.

> **Headline answer first:** yes — *everything* that matters is saved to the
> device and reloaded automatically the next time you open Termux and run
> `nm chat`. API keys, your chosen model provider, her mood, her memory, the
> relationship stage, and your chat settings all live under `~/.nomorals`
> (your Termux home directory, on the phone's storage). Minimizing, closing,
> or even rebooting the phone does **not** erase any of it. What *does* end
> when the process dies is only the live llama.cpp server and open network
> sessions — both restart cleanly from the saved config. Details in
> [§Persistence](#persistence-what-lives-where).

---

## 0. Install Termux the right way

The Play Store build of Termux is outdated and **will break** (package signing
changed). Use one of these:

- **F-Droid** (recommended): install the [Termux](https://f-droid.org/packages/com.termux/)
  app from F-Droid.
- **GitHub**: download the latest `.apk` from the
  [Termux releases](https://github.com/termux/termux-app/releases).

Open it and grant it access to the internet. That's the only permission you
need for everything in this guide except media files.

---

## 1. System packages

```bash
pkg update -y
pkg upgrade -y
pkg install -y python git cmake clang make file
```

- `python` — the runtime (the core has **zero** mandatory dependencies).
- `git` — clone the repo.
- `cmake`/`clang`/`make` — build `llama.cpp` for the local model (skip if you
  only want the cloud providers).
- `file` — used by media parsing.

Check you're on a supported Python:

```bash
python --version        # needs 3.11 or newer; Termux ships 3.12+
```

---

## 2. Get the code

```bash
cd ~
git clone https://github.com/Oluwacutyp/No-morals-ai.git
cd No-morals-ai
git checkout arena/01a088e0-no-morals-ai
```

> **Why the checkout?** `git clone` lands you on the default branch
> (`main`), which is the old skeleton — just a README. All the actual code
> lives on the working branch `arena/01a088e0-no-morals-ai`. If you cloned
> before this note existed, run `git checkout arena/01a088e0-no-morals-ai`
> inside your repo folder; it takes seconds and downloads the rest.

**Private repo? The clone will ask for a username + "password" — the
password is a Personal Access Token, not your GitHub password, and the
prompt hides what you type (on Termux it can look like a blank screen; the
input still works).** Create a token once: GitHub → Settings → Developer
settings → Personal access tokens → Tokens (classic) → `repo` scope. Then:

```bash
git config --global credential.helper store   # saves the token in ~/.git-credentials
git clone https://github.com/Oluwacutyp/No-morals-ai.git
# Username: Oluwacutyp
# Password: paste the ghp_... token (invisible, then Enter)
```

If the prompt blanks the screen entirely, skip the prompt altogether:

```bash
git clone "https://Oluwacutyp:PASTE_TOKEN@github.com/Oluwacutyp/No-morals-ai.git"
cd No-morals-ai && git remote set-url origin https://github.com/Oluwacutyp/No-morals-ai.git
git checkout arena/01a088e0-no-morals-ai   # see note above — main is just the skeleton
```

Either way, `git pull` works silently afterwards — the credential stays on
the device, like everything else in this setup.

The core runs **directly** — no `pip install` required for the first run:

```bash
python -m nomorals --help      # sanity: you should see the command list
```

Optional — get the short `nm` command instead of `python -m nomorals`:

```bash
python -m pip install -e .     # installs the `nm` entry point
nm --help
```

> Below I use `python -m nomorals` so the guide works even without the install
> step. If you ran the install, `nm …` is the same thing.

---

## 3. Build llama.cpp (only for the local-model track)

If you'll use a local GGUF model, build the server binary once. This takes a
while on a phone (10–30 min) — it's a one-time cost.

```bash
cd ~
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_ALL_INSTRUCTIONS=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j4
ls -la build/bin/llama-server     # ← this file is what the companion will spawn
cd ~/No-morals-ai
```

- `GGML_CPU_ALL_INSTRUCTIONS=ON` builds for every ARM instruction set the phone
  has (phones vary; this is the safe flag).
- `-j4` — don't use `-j$(nproc)`; phones choke on full-parallel builds.
- The setup script (`scripts/termux_setup.sh`) does steps 1–4 for you
  automatically if you'd rather not run them by hand.

**Skip this section entirely** if you're using Groq or Hugging Face instead.

---

## 4. First-run configuration

Create the config home and the `.env` that holds your API keys:

```bash
mkdir -p ~/.nomorals
cp ~/No-morals-ai/.env.example ~/.nomorals/.env
chmod 600 ~/.nomorals/.env       # keys stay private to your user
nano ~/.nomorals/.env
```

In `.env`, set at minimum:

```ini
NM_PROFILE=termux                # ← REQUIRED on a phone (see below)
```

Then **one** of the model keys, depending on your track:

| Track | Set in `.env` |
|-------|---------------|
| **A — local GGUF** | nothing (offline). Optionally `HF_TOKEN` to download a model. |
| **B — Groq** | `GROQ_API_KEY=*** |
| **C — Hugging Face** | `HF_TOKEN=*** |

Also set `NM_PROFILE=termux`. This is important: it turns off process pools
(`fork()` is unreliable on Android), lowers the context to a phone-sized 2048
for local models, and enables auto-start of the local server. Without it the
system still runs, just less tuned.

Save and close. **This file is where your API keys live on the device** — they
are read on every start and are never sent anywhere except the provider.

---

## 5. Sanity check

```bash
python -m nomorals doctor
```

You should see: `profile=termux`, a database path under `~/.nomorals`, schema
version, and a feature report. Then:

```bash
python -m nomorals models --local-doctor     # (local track only)
```

This tells you, without starting anything, whether it found the
`llama-server` binary and a `.gguf` file, and gives the exact fix for whatever
is missing. **Run this first if a local model misbehaves** — it's the
diagnostic this whole design was built around.

---

## 6. Choose your model track

Pick one. You can switch later with one command (your choice is saved).

### Track A — Local GGUF (offline, private, free after download)

```bash
# Download a model into ~/.nomorals/models (needs HF_TOKEN in .env).
python -m nomorals models --fetch dolphin-8b

# Or point at a .gguf you already have on the phone:
#   put it under ~/.nomorals/models/ and use its filename below.

# Use it:
python -m nomorals models --set-provider llama_cpp
```

The `--fetch` argument is a family name (the command tells you the exact repo
it resolved to), or an explicit `org/repo` id. Available family names — see
them all with `python -m nomorals models --catalog --kind gguf`:

| You type | You get |
|----------|---------|
| `dolphin-8b` (or `dolphin`, or `llama`) | Dolphin 2.9 on **Llama-3 8B** GGUF — uncensored Llama 8B, the Termux default |
| `mistral` | Mistral 7B Instruct GGUF |
| `qwen` | Qwen2.5 7B Instruct GGUF |
| `phi-3.5` | Phi-3.5 mini GGUF — smaller, fastest on weak phones |

The pick is the **smallest usable quant** in the repo (a Q4 for ~4–5 GB on an
8B), so the download fits a phone. (The `~16GB` number you may see in
`nm models --catalog` is the fp16 weight size for capacity planning — the
actual download is the Q4 quant, about a third of that.) Your own Llama 8B
GGUF that "fails to load": put the `.gguf` file under `~/.nomorals/models/`,
then `nm models --local-doctor` will tell you exactly what the loader sees.

On the **first** `nm chat`, the companion spawns `llama-server` itself and
*waits patiently* — an 8B on a phone takes several minutes to load, not
seconds. Watch for the "local server ready" log line before you type.

If you already built llama.cpp (step 3) and have a `.gguf`, you can also start
the server by hand to see it work:

```bash
python -m nomorals models --start-local
python -m nomorals models --local-status
python -m nomorals models --stop-local
```

### Track B — Groq (fast, no local model, free tier)

```bash
# after setting GROQ_API_KEY in ~/.nomorals/.env:
python -m nomorals models --set-provider groq
```

Instant responses, no phone load. Good way to try the companion before you've
finished building llama.cpp.

### Track C — Hugging Face (serverless or your own endpoint)

```bash
# after setting HF_TOKEN in ~/.nomorals/.env:
python -m nomorals models --set-provider hf
```

### Switching later

```bash
python -m nomorals models --set-provider llama_cpp --fallback-chain groq,mock
```

The provider **and** fallback chain are saved to the database, so the choice
survives restarts — local first, Groq if the local server is down, mock as a
last resort so you always have *a* voice.

---

## 7. The first conversation (local console)

```bash
termux-wake-lock                 # keep the CPU from sleeping (see §Persistence)
python -m nomorals chat
```

You get a console. Type to talk; `exit` closes it. Try:

```
> hey, how's your day going
> /status
> /mood tired
> how was your weekend
```

- `/status` shows her mood, relationship stage, active platforms, power mode,
  and counters.
- `/mood tired` forces a mood; `/mood energy=20 frustration=70` sets
  dimensions; `/mood reset` returns to baselines.
- These in-chat commands work on **any** platform once it's connected, not
  just the console.

This is your trail run: real mood shifts, memory writes, and persistence — all
offline if you're on Track A.

---

## 8. (Optional) Connect Telegram — the real platform

The console is for development. The point is her on Telegram as a userbot.

1. Get an API id/hash from https://my.telegram.org → *API development tools*.
2. Install the extra dependency:
   ```bash
   python -m pip install telethon
   ```
3. In `~/.nomorals/.env`:
   ```ini
   NM_CHAT_TELEGRAM_ENABLED=true
   NM_CHAT_TELEGRAM_API_ID=12345678
   NM_CHAT_TELEGRAM_API_HASH=0123456789abcdef0123456789abcdef
   NM_PARTNER_PLATFORMS=local,telegram
   NM_PARTNER_OWNER_CHATS=telegram:YOUR_TELEGRAM_ID
   ```
   (Find your numeric Telegram id by messaging @userinfobot.)
4. Run:
   ```bash
   python -m nomorals chat
   ```
   First time, a **TELEGRAM FIRST-TIME LOGIN** banner appears *before the
   console starts* — enter your number in international format (e.g.
   `234803...`), then the code Telegram sends you. It saves a **session file**
   under `~/.nomorals`. **That session file is the credential — it's saved on
   the device and reused next time, so you log in once.** (The login owns the
   keyboard first on purpose: while the console runs, it would eat the
   login's keystrokes.)

Discord and WhatsApp follow the same pattern (`.env` flag + `NM_PARTNER_PLATFORMS`)
— see the main README.

### Talking to her from the same phone (no second number needed)

A userbot *is* your account, so "you" and "she" share one identity — the
trick is to text yourself:

- **Telegram → Saved Messages.** Open the Saved Messages chat and type to
  her there. Your outgoing messages in the self-chat are treated as inbound
  from you; her replies land in the same chat. (Both sides render as your
  own bubbles — that's just how Saved Messages looks.)
- **WhatsApp → "Message Yourself".** Same pattern: open the chat with
  yourself and type. The bridge forwards self-chat messages, so this works
  from the single phone running Termux.
- **Discord** needs no trick — the bot is a separate identity, you simply
  DM her (or message her in a shared server).

For the self-chat flow, `NM_PARTNER_OWNER_CHATS` is your own id:
`telegram:<your user id>` / `whatsapp:<your number>@s.whatsapp.net`.

> Live platform logins need a real account and were not exercised in the
> development sandbox. Expect first-run friction (login code, bot-token
> intents, QR scan) and work through it live.

---

## Persistence — what lives where

This is the part that answers "will my API keys and setup survive closing the
app?" **Yes.** Everything persistent is on disk under `~/.nomorals`:

| What | Where | Survives close/reboot? |
|------|-------|------------------------|
| API keys, profile, all config | `~/.nomorals/.env` | ✅ yes |
| Chosen provider + fallback chain | SQLite `kv_store` | ✅ yes |
| Her mood (10 dimensions) + history | SQLite `mood_state` / `mood_history` | ✅ yes |
| Memory, facts, episodes, training pairs | SQLite + `data/training/*.jsonl` | ✅ yes |
| Relationship stage, trust, fights, milestones | SQLite `relationship` | ✅ yes |
| Chat registry (who's your partner, US tagging) | SQLite `chats` | ✅ yes |
| Telegram login session | `~/.nomorals/data/telegram.session` | ✅ yes (log in once) |
| Power mode unlocked state | SQLite `kv_store` | ✅ yes |
| Model weights (GGUF) | `~/.nomorals/models/` | ✅ yes |
| Backups (incl. GitHub push) | `~/.nomorals/backups/` | ✅ yes |
| **Running llama.cpp server process** | RAM | ❌ no — respawns from config |
| **Open Telegram/WebSocket sessions** | RAM | ❌ no — reconnect from saved session |

So: edit `.env` once, run `nm chat`, and every time after that the same
companion comes back with the same mood, the same memory, and the same model —
no re-entering anything.

**Keeping a *live* session running** (a different thing from persistence):
Android kills background apps aggressively. If you want her to stay up while
the screen is off:

```bash
termux-wake-lock                  # prevents CPU sleep while Termux is open
```

and in Android Settings → Apps → Termux → Battery, choose **Unrestricted**.
That's for keeping a *running* process alive; your *data* is safe either way.

---

## Cheat sheet (the whole trail run, condensed)

```bash
# 1. packages
pkg update -y && pkg upgrade -y && pkg install -y python git cmake clang make file

# 2. code
cd ~ && git clone https://github.com/Oluwacutyp/No-morals-ai.git && cd No-morals-ai

# 3. (local track) build the server
cd ~ && git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_ALL_INSTRUCTIONS=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j4 && cd ~/No-morals-ai

# 4. config
mkdir -p ~/.nomorals && cp .env.example ~/.nomorals/.env && chmod 600 ~/.nomorals/.env
nano ~/.nomorals/.env                       # set NM_PROFILE=termux + a model key

# 5. check
python -m nomorals doctor
python -m nomorals models --local-doctor    # (local track)

# 6. model — pick one
python -m nomorals models --fetch dolphin-8b && python -m nomorals models --set-provider llama_cpp
#   or:  python -m nomorals models --set-provider groq
#   or:  python -m nomorals models --set-provider hf

# 7. talk
termux-wake-lock && python -m nomorals chat
```

Or do 1–4 in one shot:

```bash
bash scripts/termux_setup.sh            # + --chat for Telegram/Discord, + --skip-llama for cloud-only
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `no llama.cpp binary found` | didn't build llama.cpp, or it's not on PATH | build (step 3); it's auto-found at `~/llama.cpp/build/bin` |
| `no .gguf found for '…'` | model not downloaded, or path points at a folder | `nm models --fetch <model>`; or point at the real file |
| `port 8080 already in use` | a stale server from a crashed run | `nm models --stop-local`, or kill the old `llama-server` |
| Local model "fails to load" after minutes | OOM on the phone | lower context: set `llm.local_ctx=1024` in `.env` (via `NM_LLM_LOCAL_CTX`), use a Q4_K_S/Q4_0 quant, or a 1.5–4B model; raise `NM_LLM_LOCAL_BOOT_TIMEOUT` if it's just slow |
| Reply is slow on first message | model still loading | wait for the "local server ready" line; it's a one-time cost per launch |
| Nothing replies, provider errors | bad/missing key, or provider down | check the key in `.env`; add a fallback: `--fallback-chain groq,mock` |
| Telegram doesn't connect | missing `telethon` / bad api id-hash | `python -m pip install telethon`; verify id/hash; re-run to re-prompt login |
| App killed while screen off | Android battery management | `termux-wake-lock` + Battery → Unrestricted (data is unaffected) |

The single most useful command when in doubt:

```bash
python -m nomorals models --local-doctor   # local track
python -m nomorals doctor                   # everything else
```
