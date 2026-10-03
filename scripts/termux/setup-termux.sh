#!/data/data/com.termux/files/usr/bin/bash
# Devon on Termux — one-shot setup
# Run: bash setup-termux.sh
set -e

echo "=== Devon Termux Setup ==="

# 1. Base packages
echo "[1/5] Installing base packages..."
pkg update -y
pkg install -y python git curl wget ffmpeg

# 2. Python dependencies (core only — no heavy ML on phone)
echo "[2/5] Installing Python deps..."
# Note: do NOT upgrade pip on Termux — it breaks the python-pip package
pip install pyyaml requests

# 3. Clone Devon (nomorals-2.0)
echo "[3/5] Cloning Devon..."
if [ -d "$HOME/nomorals-2.0" ]; then
  echo "Already exists, pulling latest..."
  cd "$HOME/nomorals-2.0" && git pull
else
  git clone https://github.com/Oluwacutyp/nomorals-2.0.git "$HOME/nomorals-2.0"
  cd "$HOME/nomorals-2.0"
fi

# 4. Install Devon (minimal — phone-friendly)
echo "[4/5] Installing Devon..."
pip install -e ".[fast]"

# 5. Create secrets template
echo "[5/5] Setting up secrets..."
if [ ! -f "$HOME/.devon-secrets" ]; then
  cat > "$HOME/.devon-secrets" << 'EOF'
# Devon secrets — fill these in
export TELEGRAM_BOT_TOKEN="your-bot-token-from-botfather"
export TELEGRAM_CHAT_ID="your-numeric-chat-id"
# Optional: for cloud LLM fallback
# export HF_TOKEN="your-huggingface-token"
# export GROQ_API_KEY="your-groq-key"
EOF
  echo "Created ~/.devon-secrets — EDIT IT with your tokens!"
else
  echo "~/.devon-secrets already exists, skipping."
fi

echo ""
echo "=== Done! ==="
echo "1. Edit ~/.devon-secrets with your Telegram bot token"
echo "2. Run: source ~/.devon-secrets && cd ~/nomorals-2.0 && python3 -m nomorals.cli tui"
echo ""
echo "For the full 3.8B local model, see docs/TERMUX_DEPLOY.md section 2-3"
