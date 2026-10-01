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
# --- brain: Hugging Face serverless Inference API ---
NM_LLM_PROVIDER=hf_serverless
NM_HF_MODEL=microsoft/Phi-3.5-mini-instruct
HF_TOKEN=hf_paste_yours_here

# --- chat: Telegram bot (long-polling, no webhook needed) ---
NM_PARTNER_PLATFORMS=telegram-bot
NM_CHAT_TELEGRAM_BOT_ENABLED=true
NM_CHAT_TELEGRAM_BOT_TOKEN=123456:ABC-paste-yours-here
NM_CHAT_TELEGRAM_BOT_CHATS=8012345678
```

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
