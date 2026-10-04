#!/bin/bash
# start_termux.sh — launch Devon on Termux: local llama-server (CodeBeast 3.8B)
# + the Telegram bot. Idempotent: re-running never duplicates processes.
set -u

REPO_DIR="$HOME/devon"
ENV_FILE="$HOME/.devon-bot.env"
LLAMA_DIR="$HOME/llama.cpp"
# Prefer CodeBeast 3.8B (the phone brain); fall back to the old dolphin 8B.
CODEBEAST="$HOME/models/Cutyp/codebeast-3.8b/merged_16bit.Q4_K_M.gguf"
DOLPHIN="$HOME/models/dolphin-8b-merged.Q4_K_M.gguf"
if [ -f "$CODEBEAST" ]; then
    MODEL="$CODEBEAST"
    CTX=8192
elif [ -f "$DOLPHIN" ]; then
    MODEL="$DOLPHIN"
    CTX=4096
else
    MODEL=""
    CTX=4096
fi
LOG_FILE="$HOME/devon-termux.log"

if [ ! -f "$ENV_FILE" ]; then
    echo "missing $ENV_FILE — see docs/TERMUX_DEPLOY.md step 5" >&2
    exit 1
fi

# shellcheck disable=SC1090
set -a
. "$ENV_FILE"
set +a

# 1) llama-server (the brain). Skip if already listening.
if ! curl -sf --max-time 3 http://localhost:8080/health >/dev/null 2>&1; then
    if [ -z "$MODEL" ]; then
        echo "no local model found. Run: nm models setup Cutyp/codebeast-3.8b" >&2
        echo "(docs/TERMUX_DEPLOY.md step 3). Starting bot WITHOUT local brain — fallbacks only." >&2
    elif [ ! -x "$LLAMA_DIR/llama-server" ]; then
        echo "llama-server not found in $LLAMA_DIR (docs/TERMUX_DEPLOY.md step 2)." >&2
        echo "Starting bot WITHOUT local brain — fallbacks only." >&2
    else
        echo "starting llama-server with $MODEL..." >&2
        # -c context; --n-gpu-layers 0 keeps it on CPU (phone GPU offload
        # is experimental — flip to 99 later if your device handles it).
        nohup "$LLAMA_DIR/llama-server" \
            -m "$MODEL" -c "$CTX" --n-gpu-layers 0 --port 8080 \
            >>"$LOG_FILE" 2>&1 &
        # wait for the model to load (takes a while on phone CPU)
        for _ in $(seq 1 60); do
            curl -sf --max-time 2 http://localhost:8080/health >/dev/null 2>&1 && break
            sleep 5
        done
        curl -sf --max-time 3 http://localhost:8080/health >/dev/null 2>&1 \
            && echo "brain online." >&2 \
            || echo "WARNING: llama-server did not come up — check $LOG_FILE" >&2
    fi
else
    echo "llama-server already running." >&2
fi

# 2) the bot itself
if pgrep -f "run_chat_bot.py" >/dev/null 2>&1; then
    echo "bot already running." >&2
    exit 0
fi

cd "$REPO_DIR" || exit 1
echo "starting devon bot..." >&2
nohup python scripts/run_chat_bot.py >>"$LOG_FILE" 2>&1 &
echo "devon is up. logs: $LOG_FILE" >&2
