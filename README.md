# No Morals AI

An autonomous AI companion with full integration across messaging, shopping, scheduling, smart home, voice, payments, and more.

## Status (September 2026)

- **98,800 lines** of Python across **278 modules**
- **1,737 tests passing** (collection green, runtime assertions being fixed)
- Active development on `arena/01a088e0-no-morals-ai` branch

## Architecture

```
nomorals/
├── agents/          # Agent system (orchestrator, planner, proactive, skills)
├── accounts/        # Credential vault, account management, sessions
├── integrations/    # External service integrations
│   ├── email        # Gmail API + IMAP/SMTP
│   ├── calendar     # Google Calendar API
│   ├── shopping     # Amazon, eBay, Walmart, Best Buy
│   ├── naija_shopping  # Jumia, Konga, Jiji, Temu, AliExpress
│   ├── price_tracker   # General-purpose (flights, GPUs, crypto, anything)
│   ├── smarthome    # Home Assistant API
│   ├── voice        # TTS (edge-tts) + STT (Whisper)
│   ├── payment      # Crypto wallets, virtual cards
│   ├── spotify      # Spotify Web API
│   ├── notion       # Notion API
│   ├── social       # Facebook, Instagram, Threads, Messenger
│   ├── plaid        # Banking (balances, transactions, liabilities)
│   └── health       # Steps, sleep, heart rate, workouts
├── scheduler/       # Cron jobs, reminders, event hooks
├── goals/           # Goal tracking with subgoals and progress
├── skills/          # Reusable playbooks
├── social/chat/     # Chat adapters (Telegram, WhatsApp, Discord, web)
├── tools/           # Agent tools (browser, shell, OSINT, etc.)
├── memory/          # Long-term memory with semantic search
├── partner/         # Companion persona, mood, relationship
├── games/           # Game engine with economy and achievements
├── books/           # AI-assisted book writing
├── training/        # Model training (Unsloth, llama-factory)
├── llm/             # LLM providers (OpenAI, local, HF)
├── storage/         # SQLite database, vectors, full-text search
├── core/            # Config, errors, HTTP, crypto, logging
└── voice/           # TTS/STT + WhatsApp/Telegram voice bridge
```

## Key Features

### Messaging
- **Telegram** (Telethon) - full bot with typing indicators, media, voice notes
- **WhatsApp** (Baileys bridge) - personal account automation
- **Discord** - bot adapter
- **Side chats** - unlimited persistent conversation threads per topic

### Shopping & Deals
- **Naija Deal Hunter** - tracks deals across Jumia, Konga, Jiji, Temu, AliExpress
- **Price Tracker** - generalized for flights, GPUs, crypto, anything with a URL
- **Steal scoring** - price vs 30-day median + cross-site comparison
- **Scheduled scans** - cron jobs for twice-daily scans + flash sale windows
- **Watchlist alerts** - notify when price drops below target

### Productivity
- **Email** - Gmail API + IMAP/SMTP (send, read, search, labels)
- **Calendar** - Google Calendar API (CRUD events, reminders)
- **Notion** - read/write pages and databases
- **Scheduler** - cron jobs, reminders with snooze, event hooks
- **Goals** - durable goals with subgoals and progress tracking
- **Skills** - reusable playbooks the bot writes for itself

### Smart Home
- **Home Assistant** - lights, thermostat, locks, scenes, automations

### Voice
- **TTS** - edge-tts (Microsoft Edge voices, free, high quality)
- **STT** - Whisper (local or API)
- **Voice notes** - send/receive on WhatsApp and Telegram
- **Voice commands** - recognize and execute

### Payments
- **Crypto** - BTC, ETH, USDT, SOL, MATIC, BNB
- **Virtual cards** - generate single-use cards for online purchases
- **Approval flow** - all payments require explicit user approval

### Finance
- **Plaid** - bank balances, transactions, recurring charges, liabilities, investments

### Health
- **Steps, sleep, heart rate, workouts** - Google Fit API + local storage

### Social Media
- **Facebook/Instagram/Threads/Messenger** - read/post/insights via Graph API
- **Spotify** - search, playback, playlists, podcasts

### Agent System
- **Planner** - natural language goal decomposition → multi-step execution
- **Proactive engine** - pattern recognition, context-aware suggestions
- **Error intelligence** - root cause analysis with fix suggestions
- **Subagents** - async delegation, coordinator fan-out
- **Memory** - long-term curated + semantic search with provenance

### Security
- **Credential vault** - AES-256 encrypted storage
- **Pre-commit secret scan** - blocks commits with leaked credentials
- **Capability policy** - audit log for all sensitive operations

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run the bot
python -m nomorals

# Run tests
pytest tests/
```

## Configuration

All config via environment variables or `~/.nomorals/.env`:

```bash
NM_LLM_PROVIDER=openrouter
NM_OPENAI_API_KEY=sk-or-v1-...
NM_TELEGRAM_BOT_TOKEN=...
NM_VAULT_PASSPHRASE=...
```

## Testing

```bash
# Full suite
pytest tests/

# Specific module
pytest tests/test_wave41.py

# With coverage
pytest tests/ --cov=nomorals
```

## License

MIT
