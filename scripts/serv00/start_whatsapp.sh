#!/bin/bash
# start_whatsapp.sh — launch the WhatsApp bridge on Serv00 (or any small host).
#
# The bridge holds the WhatsApp Web session (Baileys) and speaks JSON-lines
# to the Python WhatsAppAdapter over 127.0.0.1:8787.
# Safe to run repeatedly: if the bridge is already running this exits.
#
# FIRST RUN must be in the foreground over SSH so you can scan the QR code:
#   node ~/devon/bridge/whatsapp-bridge.mjs
# After the session is saved to bridge/.creds/, this script keeps it alive.
set -u

REPO_DIR="$HOME/devon"
BRIDGE_DIR="$REPO_DIR/bridge"
LOG_FILE="$HOME/whatsapp-bridge.log"

if [ ! -d "$BRIDGE_DIR/node_modules" ]; then
    echo "bridge deps missing — run: cd $BRIDGE_DIR && npm install" >&2
    exit 1
fi

if pgrep -f "whatsapp-bridge.mjs" >/dev/null 2>&1; then
    exit 0
fi

cd "$BRIDGE_DIR" || exit 1
# nohup so it survives the cron shell; the bridge reconnects on its own
exec nohup node whatsapp-bridge.mjs >>"$LOG_FILE" 2>&1
