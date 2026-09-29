#!/data/data/com.termux/files/usr/bin/bash
# termux_setup.sh — one-shot first-run setup for NoMorals Core on Termux.
#
# Usage (on the phone):
#   bash scripts/termux_setup.sh            # core + llama.cpp build
#   bash scripts/termux_setup.sh --chat     # + telethon/discord.py for Telegram/Discord
#   bash scripts/termux_setup.sh --skip-llama   # no local model needed (Groq/HF only)
#
# Everything persistent lands in ~/.nomorals (your home dir, survives
# Termux restarts, app close, even reboots): config, .env (API keys),
# SQLite database (mood, memory, chats, audit), and model weights.
set -euo pipefail

WITH_CHAT=0
SKIP_LLAMA=0
for arg in "$@"; do
    case "$arg" in
        --chat) WITH_CHAT=1 ;;
        --skip-llama) SKIP_LLAMA=1 ;;
        *) echo "unknown flag: $arg (use --chat / --skip-llama)"; exit 2 ;;
    esac
done

if [ -z "${PREFIX:-}" ] || [ "${PREFIX#/data/data/com.termux/files/usr}" != "${PREFIX}" ]; then
    echo "warning: this does not look like Termux (\$PREFIX=${PREFIX:-unset}) — continuing anyway."
fi

echo "── 1/5 system packages"
pkg update -y
pkg upgrade -y
pkg install -y python git cmake clang make file

echo "── 2/5 get the code"
if [ -d "$HOME/No-morals-ai" ]; then
    cd "$HOME/No-morals-ai"
    git pull --ff-only || echo "   (git pull failed — using what's on disk)"
else
    cd "$HOME"
    git clone https://github.com/Oluwacutyp/No-morals-ai.git
    cd No-morals-ai
fi
cd "$HOME/No-morals-ai"

echo "── 3/5 optional python extras (zero mandatory deps; core runs as-is)"
if [ "$WITH_CHAT" -eq 1 ]; then
    python -m pip install --quiet telethon discord.py
else
    echo "   skipping telethon/discord.py (add later: python -m pip install telethon discord.py)"
fi

echo "── 4/5 build llama.cpp (llama-server for local GGUF models)"
if [ "$SKIP_LLAMA" -eq 0 ]; then
    cd "$HOME"
    if [ ! -d llama.cpp ]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp
    fi
    cd llama.cpp
    # GGML_CPU_ALL_INSTRUCTIONS=ON: phone CPUs vary; build for every ARM set.
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_ALL_INSTRUCTIONS=ON -DLLAMA_CURL=OFF
    # -j4: phones choke on -j$(nproc); 4 is the sweet spot.
    cmake --build build --config Release -j4
    ls -la build/bin/llama-server
    cd "$HOME/No-morals-ai"
else
    echo "   skipped (--skip-llama)"
fi

echo "── 5/5 first-run configuration (~/.nomorals)"
mkdir -p "$HOME/.nomorals"
ENV_FILE="$HOME/.nomorals/.env"
if [ ! -f "$ENV_FILE" ]; then
    sed -e 's/^NM_PROFILE=workstation/NM_PROFILE=termux/' \
        -e 's/^NM_CHAT_LOCAL_ENABLED=true/NM_CHAT_LOCAL_ENABLED=true/' \
        "$HOME/No-morals-ai/.env.example" > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "created $ENV_FILE (perms 600) — EDIT IT for API keys"
else
    echo "keeping existing $ENV_FILE"
fi

echo
echo "═══════════════════════════════════════════════════════════════"
echo "Setup done. Next:"
echo
echo "  1. edit your keys:        nano ~/.nomorals/.env"
echo "  2. sanity check:          python -m nomorals doctor"
echo "  3. local model (pick one):"
echo "       python -m nomorals models --fetch dolphin-8b   (needs HF_TOKEN)"
echo "       python -m nomorals models --set-provider llama_cpp"
echo "     …or cloud fallback:"
echo "       (set GROQ_API_KEY in .env) python -m nomorals models --set-provider groq"
echo "  4. talk:                  python -m nomorals chat"
echo
echo "Keep Termux awake while chatting:  termux-wake-lock"
echo "Everything in ~/.nomorals survives app close, minimize, and reboot."
echo "═══════════════════════════════════════════════════════════════"
