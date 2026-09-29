# Telegram Setup Guide

Your bot is currently only running on the local console. To use `/game` commands in Telegram chats, you need to connect it to Telegram.

## Step 1: Get Telegram API Credentials

1. Go to https://my.telegram.org/apps
2. Log in with your phone number
3. Click "API development tools"
4. Create a new application (or use existing one)
5. Note down:
   - **API ID** (integer, like `12345678`)
   - **API Hash** (string, like `abcdef1234567890`)

## Step 2: Configure nomorals

Edit `~/.nomorals/config.toml` and add:

```toml
[chat]
telegram_enabled = true
telegram_api_id = 12345678  # Your API ID (integer, no quotes)
telegram_api_hash = "abcdef1234567890"  # Your API hash (string with quotes)
telegram_session = "data/telegram.session"
telegram_chats = ""  # Empty = allow all chats

[partner]
platforms = "telegram,local"  # Add telegram to the list
```

## Step 3: First-Time Login

Run the bot for the first time:

```bash
cd ~/No-morals-ai
nm chat start
```

The first time, it will ask for:
1. Your phone number (in international format, e.g., `+2348031234567`)
2. The login code Telegram sends you
3. Your 2FA password (if you have one)

After successful login, the session is saved to `data/telegram.session`.

## Step 4: Test It

1. Open Telegram on your phone
2. Send a message to yourself (Saved Messages) or a friend
3. Type `/game` - the bot should respond

## Troubleshooting

### "telegram unavailable" error
- Check that `telethon` is installed: `pip install telethon`
- Verify API credentials are correct in config.toml

### Bot doesn't respond in Telegram
- Check logs: `tail -f ~/.nomorals/logs/nomorals.log | grep telegram`
- Look for "telegram: DELIVERING inbound" messages
- If you see "ignoring message" → chat ID is not in allowlist

### Bot responds in console but not Telegram
- Make sure `platforms` includes "telegram": `platforms = "telegram,local"`
- Restart the bot: `nm chat stop && nm chat start`

## How It Works

The Telegram adapter uses a **userbot** (your own Telegram account), not a separate bot account. This means:

- The bot can read and send messages in any chat you're in
- It appears as YOU sending messages (be careful!)
- It can access DMs, groups, and channels
- It uses your account's privacy settings

**Important:** Since it's your account, the bot will send messages as you. Don't use it in chats where you don't want automated messages appearing from your account.

## Security Notes

- Keep `data/telegram.session` secure - it's like a password
- Don't commit it to git
- If compromised, delete it and re-login
- Consider using a separate Telegram account for the bot

## Next Steps

Once Telegram is working:
- Test `/game` in a DM with yourself
- Test `/game` in a DM with a friend
- Test `/game` in a group chat
- Check logs for any errors

If `/game` still doesn't work after setup, check the logs and share the error message.
