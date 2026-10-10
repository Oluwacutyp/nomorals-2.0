#!/bin/bash
# Devon Secrets Setup — run once on a fresh machine.
# Creates ~/.devon-secrets/ with a file per secret.
# Fill in your values, then: chmod 600 ~/.devon-secrets/*
#
# Usage: bash setup-secrets.sh
# Then edit each file with your real key.

SECRETS_DIR="$HOME/.devon-secrets"
mkdir -p "$SECRETS_DIR"

# Format: "FILENAME|Description|Required?"
declare -a SECRETS=(
  # --- Core / Chat ---
  "TELEGRAM_BOT_TOKEN|Telegram bot token from @BotFather|yes"
  "GROQ_API_KEY|Groq API key (LLM fallback)|yes"
  "OPENROUTER_API_KEY|OpenRouter key (model failover)|no"
  "HF_TOKEN|HuggingFace token (model downloads)|yes"

  # --- Trading ---
  "EXNESS_API_KEY|Exness API key|no"
  "EXNESS_PRIVATE_KEY|Exness Ed25519 private key|no"
  "EXNESS_ACCOUNT_ID|Exness account ID|no"
  "BINANCE_API_KEY|Binance (crypto data)|no"
  "BINANCE_API_SECRET|Binance secret|no"

  # --- Search & Research ---
  "TAVILY_API_KEY|Tavily search|no"
  "SERPER_API_KEY|Serper search|no"
  "EXA_API_KEY|Exa search|no"
  "BRAVE_SEARCH_API_KEY|Brave search|no"

  # --- Media / AI Generation ---
  "LEONARDO_API_KEY|Leonardo image gen|no"
  "KLING_API_KEY|Kling video gen|no"
  "MINIMAX_API_KEY|MiniMax video gen|no"
  "NANO_BANANA_API_KEY|Nano Banana image gen|no"
  "GOOGLE_FLOW_API_KEY|Google Flow video|no"
  "GEMINI_API_KEY|Google Gemini|no"
  "ANTHROPIC_API_KEY|Anthropic Claude|no"
  "DEEPSEEK_API_KEY|DeepSeek|no"
  "OPENAI_API_KEY|OpenAI|no"

  # --- Finance Data ---
  "ALPHA_VANTAGE_API_KEY|Alpha Vantage stocks|no"
  "FINNHUB_API_KEY|Finnhub stocks|no"

  # --- Social ---
  "TWITTER_API_KEY|X/Twitter|no"
  "DISCORD_BOT_TOKEN|Discord bot|no"

  # --- Utilities ---
  "CAPTCHA_API_KEY|2Captcha / CapSolver|no"
  "DUFFEL_API_KEY|Duffel flights|no"
  "FOOTBALL_DATA_API_KEY|Football data|no"
)

echo "Setting up secrets in $SECRETS_DIR"
echo ""

for entry in "${SECRETS[@]}"; do
  IFS='|' read -r filename desc required <<< "$entry"
  filepath="$SECRETS_DIR/$filename"

  if [ -f "$filepath" ]; then
    echo "  SKIP  $filename (already exists)"
  else
    echo "# $desc" > "$filepath"
    echo "# Required: $required" >> "$filepath"
    echo "# Replace this line with your actual key" >> "$filepath"
    echo "REPLACE_ME" >> "$filepath"
    chmod 600 "$filepath"
    echo "  CREATED $filename"
  fi
done

echo ""
echo "Done. Now edit each file in $SECRETS_DIR and replace REPLACE_ME with your real key."
echo "Example: nano $SECRETS_DIR/TELEGRAM_BOT_TOKEN"
echo ""
echo "To load all secrets into your shell:"
echo "  for f in $SECRETS_DIR/*; do export \$(basename \$f)=\$(grep -v '^#' \$f | head -1); done"
