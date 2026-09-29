#!/usr/bin/env python3
"""Diagnose Telegram chat allowlist issues.

Run this to see which chats are allowed and why /game might not work.
"""
from nomorals.core.config import load_settings

settings = load_settings()
telegram_chats = settings.chat.telegram_chats

print("=" * 70)
print("TELEGRAM CHAT ALLOWLIST DIAGNOSIS")
print("=" * 70)
print()

if not telegram_chats or not telegram_chats.strip():
    print("✓ telegram_chats is EMPTY → ALL chats are allowed")
    print()
    print("This means the bot should respond to /game commands in ANY Telegram chat.")
    print()
    print("If /game still doesn't work, the issue is NOT the allowlist.")
    print("Check:")
    print("  1. Is the Telegram adapter running? (nm status)")
    print("  2. Is the chat a DM or group? (both should work)")
    print("  3. Are there errors in the logs? (~/.nomorals/logs/nomorals.log)")
else:
    chats = [c.strip() for c in telegram_chats.split(',') if c.strip()]
    print(f"✗ telegram_chats is SET → Only {len(chats)} chat(s) allowed:")
    print()
    for i, chat in enumerate(chats, 1):
        print(f"  {i}. {chat}")
    print()
    print("If your friend's chat ID is NOT in this list, /game won't work there.")
    print()
    print("FIX OPTIONS:")
    print()
    print("Option 1: Allow ALL chats (recommended for testing)")
    print("  Edit ~/.nomorals/config.toml and set:")
    print('    telegram_chats = ""')
    print()
    print("Option 2: Add your friend's chat ID to the allowlist")
    print("  Find the chat ID (it's in the bot logs when a message arrives)")
    print("  Then add it to telegram_chats:")
    print('    telegram_chats = "123456789,987654321"')
    print()
    print("After changing config, restart the bot:")
    print("  nm chat stop")
    print("  nm chat start")

print()
print("=" * 70)
print("HOW TO FIND A CHAT ID")
print("=" * 70)
print()
print("1. Send a message in the Telegram chat where /game doesn't work")
print("2. Check the logs:")
print("   tail -f ~/.nomorals/logs/nomorals.log | grep 'telegram:'")
print("3. Look for a line like:")
print("   telegram: DELIVERING inbound from <name> in <CHAT_ID>: <message>")
print("4. The <CHAT_ID> is what you need to add to telegram_chats")
print()
print("=" * 70)
