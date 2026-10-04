#!/bin/bash
# start_termux.sh — launch Devon on Termux: local llama-server (CodeBeast 3.8B)
# + the Telegram bot. Idempotent: re-running never duplicates processes.
#
# Usage:
#   bash start_termux.sh            # start brain + bot + watchdog (detached)
#   bash start_termux.sh watchdog   # run the watchdog in the foreground
#                                   # (the script launches this itself, detached)
#
# The watchdog keeps the bot (and llama-server) alive: if the process dies
# — OOM-killed, uncaught crash, phone doze — it is restarted with
# exponential backoff. Without this, one crash = a dead bot until the owner
# notices and re-runs the script by hand.
set -u

# The setup script clones into ~/nomorals-2.0; older notes say ~/devon.
# Prefer the real checkout, whichever name it has.
REPO_DIR=""
for candidate in "$HOME/nomorals-2.0" "$HOME/devon"; do
    if [ -d "$candidate/scripts/run_chat_bot.py" ]; then
        REPO_DIR="$candidate"
        break
    fi
done
if [ -z "$REPO_DIR" ]; then
    echo "no devon checkout found (looked for ~/nomorals-2.0 and ~/devon)" >&2
    echo "clone it first: git clone https://github.com/Oluwacutyp/nomorals-2.0.git ~/nomorals-2.0" >&2
    exit 1
fi

# Secrets: the deploy doc uses ~/.devon-bot.env; setup-termux.sh writes
# ~/.devon-secrets. Accept either.
ENV_FILE=""
for candidate in "$HOME/.devon-bot.env" "$HOME/.devon-secrets"; do
    if [ -f "$candidate" ]; then
        ENV_FILE="$candidate"
        break
    fi
done
if [ -z "$ENV_FILE" ]; then
    echo "missing secrets file — create ~/.devon-bot.env (see docs/TERMUX_DEPLOY.md step 5)" >&2
    echo "or run: bash scripts/termux/setup-termux.sh" >&2
    exit 1
fi

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
BOT_PATTERN="scripts/run_chat_bot.py"
# Absolute path to this script, so the detached watchdog re-invokes the
# right file no matter what cwd it was launched from.
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

log() { echo "$(date '+%F %T') [termux] $*" >>"$LOG_FILE"; }

start_brain() {
    # Returns 0 if the brain is up (or intentionally skipped), 1 if it
    # failed to come up (watchdog will retry later).
    if curl -sf --max-time 3 http://localhost:8080/health >/dev/null 2>&1; then
        return 0
    fi
    if [ -z "$MODEL" ]; then
        log "no local model found — bot runs WITHOUT local brain (fallbacks only)"
        return 0
    fi
    if [ ! -x "$LLAMA_DIR/llama-server" ]; then
        log "llama-server missing in $LLAMA_DIR — bot runs WITHOUT local brain"
        return 0
    fi
    log "starting llama-server with $MODEL"
    # -c context; --n-gpu-layers 0 keeps it on CPU (phone GPU offload
    # is experimental).
    nohup "$LLAMA_DIR/llama-server" \
        -m "$MODEL" -c "$CTX" --n-gpu-layers 0 --port 8080 \
        >>"$LOG_FILE" 2>&1 &
    for _ in $(seq 1 60); do
        curl -sf --max-time 2 http://localhost:8080/health >/dev/null 2>&1 && break
        sleep 5
    done
    if curl -sf --max-time 3 http://localhost:8080/health >/dev/null 2>&1; then
        log "brain online"
        return 0
    fi
    log "WARNING: llama-server did not come up — check $LOG_FILE"
    return 1
}

start_bot() {
    if pgrep -f "$BOT_PATTERN" >/dev/null 2>&1; then
        return 0
    fi
    log "starting devon bot (repo: $REPO_DIR, env: $ENV_FILE)"
    # shellcheck disable=SC1090
    ( set -a; . "$ENV_FILE"; set +a; cd "$REPO_DIR" && \
      nohup python scripts/run_chat_bot.py >>"$LOG_FILE" 2>&1 & )
    # give it a moment; the watchdog loop re-checks anyway
    sleep 3
    if pgrep -f "$BOT_PATTERN" >/dev/null 2>&1; then
        log "bot process running"
        return 0
    fi
    log "WARNING: bot process did not stay up — check $LOG_FILE"
    return 1
}

watchdog() {
    # Runs forever (detached): keeps the brain and the bot alive.
    log "watchdog started (pid $$)"
    bot_backoff=10
    brain_backoff=30
    while true; do
        if ! pgrep -f "$BOT_PATTERN" >/dev/null 2>&1; then
            log "watchdog: bot is down — restarting in ${bot_backoff}s"
            sleep "$bot_backoff"
            if start_bot; then
                bot_backoff=10
            else
                bot_backoff=$(( bot_backoff < 300 ? bot_backoff * 2 : 300 ))
            fi
        else
            bot_backoff=10
        fi
        if ! curl -sf --max-time 3 http://localhost:8080/health >/dev/null 2>&1; then
            # Only restart the brain when we have a model to serve; missing
            # model/server is a setup problem, not a crash.
            if [ -n "$MODEL" ] && [ -x "$LLAMA_DIR/llama-server" ]; then
                log "watchdog: brain is down — restarting in ${brain_backoff}s"
                sleep "$brain_backoff"
                # make sure a half-dead server isn't holding the port
                pkill -f "llama-server.*--port 8080" 2>/dev/null || true
                if start_brain; then
                    brain_backoff=30
                else
                    brain_backoff=$(( brain_backoff < 600 ? brain_backoff * 2 : 600 ))
                fi
            fi
        else
            brain_backoff=30
        fi
        sleep 30
    done
}

if [ "${1:-}" = "watchdog" ]; then
    watchdog
    exit 0
fi

# shellcheck disable=SC1090
set -a
. "$ENV_FILE"
set +a

start_brain || true

if pgrep -f "$BOT_PATTERN" >/dev/null 2>&1; then
    echo "bot already running." >&2
else
    cd "$REPO_DIR" || exit 1
    echo "starting devon bot..." >&2
    nohup python scripts/run_chat_bot.py >>"$LOG_FILE" 2>&1 &
    echo "devon is up. logs: $LOG_FILE" >&2
fi

# The watchdog: one instance, detached. It restarts the bot (and the brain)
# if they ever die.
if pgrep -f "start_termux.sh watchdog" >/dev/null 2>&1; then
    echo "watchdog already running." >&2
else
    echo "starting watchdog..." >&2
    nohup bash "$SCRIPT_PATH" watchdog >>"$LOG_FILE" 2>&1 &
    echo "watchdog armed. logs: $LOG_FILE" >&2
fi
