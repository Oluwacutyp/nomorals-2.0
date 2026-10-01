#!/bin/bash
# start_bot.sh — launch the Devon Telegram bot on Serv00 (or any small host).
#
# Reads secrets from ~/.devon-bot.env (chmod 600), then execs the launcher.
# Safe to run repeatedly: if the bot is already running this script exits.
set -u

REPO_DIR="$HOME/devon"
ENV_FILE="$HOME/.devon-bot.env"
PYTHON="$HOME/bot-venv/bin/python"
LOG_FILE="$HOME/bot.log"

if [ ! -f "$ENV_FILE" ]; then
    echo "missing $ENV_FILE — copy the template from docs/SERV00_DEPLOY.md" >&2
    exit 1
fi

if pgrep -f "run_chat_bot.py" >/dev/null 2>&1; then
    exit 0
fi

# shellcheck disable=SC1090
set -a
. "$ENV_FILE"
set +a

cd "$REPO_DIR" || exit 1
exec "$PYTHON" scripts/run_chat_bot.py >>"$LOG_FILE" 2>&1
