# Running the Devon Telegram bot on Serv00 (free, no card)

**The shape:** your bot lives on Serv00's free shell host; its *brain* is your
fine-tuned model served by Hugging Face's free serverless Inference API. Total
cost: $0. No credit card anywhere in the chain.

```
Telegram user  →  Serv00 (Devon bot, long-polling)  →  HF Inference API (your model)
```

## What you need (all free)

1. **Telegram bot token** — message [@BotFather](https://t.me/BotFather) on
   Telegram, send `/newbot`, follow the prompts. It replies with a token like
   `123456:ABC-DEF...`. Keep it private.
2. **Your chat id** — message [@userinfobot](https://t.me/userinfobot), it
   replies with your numeric id (e.g. `8012345678`). This locks the bot to you.
3. **Hugging Face token** — [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   → "Create new token", type **Read**. Copy the `hf_...` value.
4. **Model repo id** — until your fine-tune finishes, use the base model:
   `microsoft/Phi-3.5-mini-instruct`. After training, switch to your repo
   (e.g. `<your-username>/codebeast-3.8b`).

## Step 1 — Create the Serv00 account

1. Go to [serv00.com](https://www.serv00.com), click **Sign up**, choose the
   **free** plan, confirm your email.
2. Log in to the panel and note your **SSH host** (something like
   `s12.serv00.com`) and username.
3. If signup says registrations are paused, wait a day or two and retry.

## Step 2 — SSH in and set up Python

From any terminal (on Android: Termux works):

```bash
ssh <username>@<ssh-host>.serv00.com
python3 --version        # need 3.11+
python3 -m venv ~/bot-venv
```

## Step 3 — Get the bot code

```bash
git clone https://github.com/Oluwacutyp/nomorals-2.0.git ~/devon
chmod +x ~/devon/scripts/serv00/start_bot.sh ~/devon/scripts/run_chat_bot.py
```

The bot's only hard dependencies for this setup are the Python standard
library — no heavy packages to install.

## Step 4 — Secrets file

Create `~/.devon-bot.env` (note the leading dot) with **your real values**:

```bash
# --- brain: Hugging Face serverless Inference API (primary) ---
NM_LLM_PROVIDER=hf_serverless
NM_HF_MODEL=SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored
HF_TOKEN=hf_paste_yours_here

# --- brain fallbacks: used automatically when HF errors or its $0.10/mo
# --- free credits run out. You only need the ones you have keys for.
NM_LLM_FALLBACK_CHAIN=groq,openrouter
NM_GROQ_API_KEY=gsk_paste_yours_here
NM_GROQ_MODEL=openai/gpt-oss-120b
NM_OPENROUTER_API_KEY=sk-or-paste_yours_here
NM_OPENROUTER_MODEL=qwen/qwen3-8b:free

# --- chat: Telegram bot (long-polling, no webhook needed) ---
NM_PARTNER_PLATFORMS=telegram-bot
NM_CHAT_TELEGRAM_BOT_ENABLED=true
NM_CHAT_TELEGRAM_BOT_TOKEN=123456:ABC-paste-yours-here
NM_CHAT_TELEGRAM_BOT_CHATS=8012345678
```

Notes on the brain section:
- `NM_HF_MODEL` is an uncensored (abliterated) Phi-3.5-mini in safetensors
  format — no training needed, ready today. Before deploying, verify HF
  actually serves it with this 10-second test (replace the token and model):
  ```bash
  curl -s -o /dev/null -w "%{http_code}\n" \
    -H "Authorization: Bearer hf_paste_yours_here" \
    https://router.huggingface.co/hf-inference/models/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored
  ```
  `200` = she can talk through it. Anything else (404/410) = HF doesn't
  serve that model — fall back to `microsoft/Phi-3.5-mini-instruct` (the
  stock, censored one) until your own `Cutyp/codebeast-3.8b` is trained.
- **Groq key (free, no card):** [console.groq.com](https://console.groq.com)
  → sign in → API Keys → Create. Free tier is generous (30 req/min).
  Note: Groq only hosts mainstream models (no uncensored ones) — it's a
  capability fallback, not a personality fallback.
- **OpenRouter key (free models available):** [openrouter.ai](https://openrouter.ai)
  → sign in → Keys → Create. Pick any `:free`-suffixed model for
  `NM_OPENROUTER_MODEL` from [openrouter.ai/models](https://openrouter.ai/models)
  (e.g. `qwen/qwen3-8b:free`); free models get 50 requests/day.
- Leave a fallback's lines out entirely if you don't have its key — the
  bot skips unconfigured providers quietly instead of failing.

Then lock it down:

```bash
chmod 600 ~/.devon-bot.env
```

Never commit this file, never paste the tokens into a chat.

## Step 5 — First run (foreground, to verify)

```bash
bash ~/devon/scripts/serv00/start_bot.sh
# in another SSH session:
tail -f ~/bot.log
```

You should see `adapters started: ['telegram-bot', ...]`. Now message your bot
on Telegram — it should reply. (First reply can be slow: HF cold-starts the
model, ~30–60s. Later replies are fast.)

Stop the foreground run with `Ctrl+C` once it works.

## Step 6 — Keep it alive with cron

```bash
crontab -e
```

Add this line (all one line):

```
* * * * * /bin/bash $HOME/devon/scripts/serv00/start_bot.sh
```

Every minute cron checks whether the bot is running and restarts it if not.
Survives Serv00 maintenance reboots on its own.

## Step 7 — Point it at YOUR model after training

Once `Cutyp/codebeast-3.8b` is on your Hugging Face account:

```bash
# edit ~/.devon-bot.env:
NM_HF_MODEL=Cutyp/codebeast-3.8b
# then restart:
pkill -f run_chat_bot.py   # cron brings it back within a minute
```

## Optional: also connect your personal Telegram account (userbot)

This gives Devon full access as *you* over MTProto: DMs, groups, channels.
Handy trick: message your own **Saved Messages** and she replies there.

**Read this first, it's important:**
- The session file (below) is a *live login* to your Telegram account —
  anyone holding it **is you** on Telegram. Serv00 is a shared host. Only
  do this if you're comfortable with that risk.
- Telegram limits or bans userbot accounts that behave like spam bots.
  Devon has per-platform rate limits, but keep proactive/auto messaging
  modest — don't let her DM strangers.

Steps:

1. **API credentials (free):** go to [my.telegram.org](https://my.telegram.org),
   log in with your phone number → **API development tools** → create an app
   → copy **api_id** and **api_hash**.
2. **Install Telethon** in the venv:
   ```bash
   ~/bot-venv/bin/pip install telethon
   ```
3. **Add to `~/.devon-bot.env`:**
   ```bash
   NM_PARTNER_PLATFORMS=telegram,telegram-bot
   NM_CHAT_TELEGRAM_ENABLED=true
   NM_CHAT_TELEGRAM_API_ID=12345678
   NM_CHAT_TELEGRAM_API_HASH=paste_yours_here
   NM_CHAT_TELEGRAM_SESSION=~/.devon-telegram.session
   # optional: NM_CHAT_TELEGRAM_CHATS=...  (limit which chats she listens in)
   ```
4. **First login is interactive** — run in the foreground over SSH (not via
   cron). She'll ask for your phone number in international format
   (e.g. `234803...`), then the code Telegram sends you, then your 2FA
   password if you have one. After that the session file is the credential
   and every later boot is silent.
5. **Lock down the session file:**
   ```bash
   chmod 600 ~/.devon-telegram.session
   ```
   Never commit it, never upload it anywhere.

## Optional: connect WhatsApp (your own account, via a bridge)

WhatsApp has no free official API for a personal account, so Devon talks to
it through a small **bridge** (`bridge/whatsapp-bridge.mjs`): a Node.js
program that holds your WhatsApp Web session (like WhatsApp Web in a
browser) and passes messages to the Python bot over your own machine only.
Nothing leaves the Serv00 box.

**Read this first:**
- The bridge saves its login in `bridge/.creds/` — that folder is a *live
  login* to your WhatsApp, same as the Telegram session file. `chmod 700`
  it, never copy it anywhere.
- WhatsApp can limit accounts that behave like spam bots. Keep auto-messaging
  modest; don't let her message strangers.

### Step A — Install Node.js and the bridge's dependencies

SSH into Serv00 and run:

```bash
node --version          # you need v18 or newer
cd ~/devon/bridge
npm install             # installs the Baileys WhatsApp library
```

If `node` is missing or too old, check Serv00's docs/forum for enabling a
newer Node — most Serv00 accounts already have one.

### Step B — Link your WhatsApp (one time, in the foreground)

This is the only interactive part. Run the bridge **directly** (not via the
script, not via cron) so you can see the QR code:

```bash
node ~/devon/bridge/whatsapp-bridge.mjs
```

You'll see a big QR code printed in the terminal. On your phone:

1. Open WhatsApp → **Settings** → **Linked Devices**
2. Tap **Link a Device** → point the camera at the QR code in the terminal

Within seconds the bridge prints `connected as 23480...@s.whatsapp.net`.
The login is now saved in `~/devon/bridge/.creds/` — every future start is
silent, no QR needed again. Lock it down and stop the foreground bridge:

```bash
chmod 700 ~/devon/bridge/.creds
# press Ctrl+C to stop the foreground bridge
```

If the QR expires before you scan (about a minute), a fresh one prints
automatically. If you ever get logged out, delete `~/devon/bridge/.creds/`
and redo this step.

### Step C — Tell the bot about WhatsApp

Add to `~/.devon-bot.env`:

```bash
NM_PARTNER_PLATFORMS=telegram,telegram-bot,whatsapp
NM_CHAT_WHATSAPP_ENABLED=true
# NM_CHAT_WHATSAPP_HOST=127.0.0.1   # defaults are fine — bridge runs locally
# NM_CHAT_WHATSAPP_PORT=8787
```

Make the start script executable:

```bash
chmod +x ~/devon/scripts/serv00/start_whatsapp.sh
```

### Step D — Start the bridge and the bot

```bash
bash ~/devon/scripts/serv00/start_whatsapp.sh   # keeps the bridge alive
bash ~/devon/scripts/serv00/start_bot.sh        # the bot itself
```

Watch both logs to confirm:

```bash
tail -5 ~/whatsapp-bridge.log   # want: "connected as ..."
tail -5 ~/bot.log               # want: adapters started including 'whatsapp'
```

Send your own WhatsApp account a message — she should reply.

### Step E — Keep the bridge alive with cron

Add a second cron line next to the bot's:

```cron
* * * * * /bin/bash $HOME/devon/scripts/serv00/start_bot.sh
* * * * * /bin/bash $HOME/devon/scripts/serv00/start_whatsapp.sh
```

Both scripts exit immediately if their program is already running, so cron
just acts as a watchdog.

### WhatsApp troubleshooting

| Symptom | Check |
|---|---|
| `whatsapp bridge not reachable at 127.0.0.1:8787` | the bridge isn't running — start it (Step D) |
| QR never appears | phone has no internet, or the bridge can't reach WhatsApp's servers |
| `connected` then drops every few minutes | network flapping — the bridge auto-reconnects; check `~/whatsapp-bridge.log` |
| Logged out, QR keeps reappearing | delete `~/devon/bridge/.creds/` and redo Step B |
| Bot sees WhatsApp but never replies | `NM_PARTNER_PLATFORMS` must include `whatsapp` and the bot must be restarted |

## Limits to respect

- **RAM:** 512 MB per process. The bot idles small (~50 MB), but don't run
  heavy commands (video rendering, big downloads) — Serv00 suspends accounts
  for CPU abuse.
- **HF free tier:** monthly credits + cold starts. Fine for personal use;
  don't publish the bot to the public on this tier.
- **Disk:** 3 GB. The repo + logs fit easily. Watch `~/bot.log` size;
  truncate it occasionally: `: > ~/bot.log`.

## Troubleshooting

| Symptom | Check |
|---|---|
| `adapters started: ['local']` only | `NM_CHAT_TELEGRAM_BOT_TOKEN` missing/wrong in `~/.devon-bot.env` |
| Bot replies with errors about the model | `HF_TOKEN` wrong, or `NM_HF_MODEL` repo id misspelled / not public |
| First reply very slow, then fine | Normal — HF cold start on the free tier |
| `telegram unavailable: No module named 'telethon'` | `~/bot-venv/bin/pip install telethon`, then restart |
| Login code asked on every boot | delete `~/.devon-telegram.session` and redo the one-time interactive login |
| `python3 --version` < 3.11 | Check Serv00 docs for a newer python, or ask in their forum |
| Everything dies hourly | CPU-abuse suspension — lighten what the bot is doing |

## Updating the bot later

```bash
cd ~/devon && git pull
pkill -f run_chat_bot.py    # cron restarts it with the new code
```
