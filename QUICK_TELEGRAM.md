# Quick Telegram Setup (3 Steps)

## 1. Get API Credentials

Go to: **https://my.telegram.org/apps**

- Log in with your phone number
- Click "API development tools"
- Create an app (any name/description)
- Copy the **API ID** (numbers) and **API Hash** (letters/numbers)

## 2. Configure

Edit `~/.nomorals/config.toml`:

```toml
[telegram]
enabled = true
api_id = 12345678
api_hash = "your_hash_here"
```

## 3. Start Bot

```bash
nm chat start
```

First time it asks for your phone number and a code Telegram sends you.

## That's It!

Now open any DM chat and type `/game`. The bot responds.

**By default, the bot works in ALL your DM chats** - no allowlist needed.

---

## Testing

In any DM chat:
```
/game
```

Should show the game menu.

```
/game 2048
```

Should start 2048.

## Troubleshooting

**"telegram: DROPPED inbound" in logs?**
- Message has no text (stickers, etc.)

**"ignoring message... not in NM_CHAT_TELEGRAM_CHATS"?**
- You set an allowlist but didn't include this chat
- Solution: Remove the allowlist or add the chat ID

**Nothing happens?**
- Check logs: `tail -f ~/.nomorals/logs/*.log | grep telegram`
- Look for "DELIVERING inbound" - that means the message reached the bot
