#!/usr/bin/env python3
"""Interactive Telegram setup for nomorals.

This script helps you configure Telegram API credentials and enable the adapter.
"""
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("TELEGRAM SETUP FOR NOMORALS")
    print("=" * 70)
    print()
    print("This will configure your bot to work with Telegram.")
    print()
    
    # Step 1: Get API credentials
    print("STEP 1: Get Telegram API Credentials")
    print("-" * 70)
    print()
    print("1. Go to: https://my.telegram.org/apps")
    print("2. Log in with your phone number")
    print("3. Click 'API development tools'")
    print("4. Create a new application (or use existing)")
    print()
    
    api_id = input("Enter your API ID (integer): ").strip()
    if not api_id.isdigit():
        print("ERROR: API ID must be an integer")
        sys.exit(1)
    
    api_hash = input("Enter your API Hash (string): ").strip()
    if not api_hash:
        print("ERROR: API Hash cannot be empty")
        sys.exit(1)
    
    print()
    print("✓ Credentials received")
    print()
    
    # Step 2: Update config
    print("STEP 2: Update Configuration")
    print("-" * 70)
    print()
    
    config_path = Path.home() / ".nomorals" / "config.toml"
    
    if not config_path.exists():
        print(f"Creating config at {config_path}")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_content = ""
    else:
        print(f"Reading existing config from {config_path}")
        config_content = config_path.read_text()
    
    # Check if telegram section exists
    if "[chat]" not in config_content:
        config_content += "\n[chat]\n"
    
    # Update telegram settings
    lines = config_content.split('\n')
    new_lines = []
    in_chat_section = False
    telegram_keys = {
        'telegram_enabled', 'telegram_api_id', 'telegram_api_hash',
        'telegram_session', 'telegram_chats'
    }
    
    for line in lines:
        if line.strip() == "[chat]":
            in_chat_section = True
            new_lines.append(line)
            continue
        
        if line.strip().startswith("[") and in_chat_section:
            in_chat_section = False
        
        # Skip existing telegram keys in chat section
        if in_chat_section and any(line.strip().startswith(f"{key} =") for key in telegram_keys):
            continue
        
        new_lines.append(line)
    
    # Add telegram config
    telegram_config = f"""
telegram_enabled = true
telegram_api_id = {api_id}
telegram_api_hash = "{api_hash}"
telegram_session = "data/telegram.session"
telegram_chats = ""  # Empty = allow all chats
"""
    
    # Insert after [chat]
    chat_index = None
    for i, line in enumerate(new_lines):
        if line.strip() == "[chat]":
            chat_index = i
            break
    
    if chat_index is not None:
        new_lines.insert(chat_index + 1, telegram_config)
    
    # Update platforms
    platforms_updated = False
    for i, line in enumerate(new_lines):
        if line.strip().startswith("platforms ="):
            # Parse existing platforms
            current = line.split("=")[1].strip().strip('"').strip("'")
            platforms = [p.strip() for p in current.split(",") if p.strip()]
            if "telegram" not in platforms:
                platforms.append("telegram")
            new_lines[i] = f'platforms = "{",".join(platforms)}"'
            platforms_updated = True
            break
    
    if not platforms_updated:
        # Add platforms to partner section
        if "[partner]" not in '\n'.join(new_lines):
            new_lines.append("\n[partner]")
        partner_index = None
        for i, line in enumerate(new_lines):
            if line.strip() == "[partner]":
                partner_index = i
                break
        if partner_index is not None:
            new_lines.insert(partner_index + 1, 'platforms = "telegram,local"')
    
    # Write config
    config_content = '\n'.join(new_lines)
    config_path.write_text(config_content)
    
    print(f"✓ Config updated at {config_path}")
    print()
    
    # Step 3: Next steps
    print("STEP 3: Next Steps")
    print("-" * 70)
    print()
    print("Configuration complete! Now:")
    print()
    print("1. Start the bot:")
    print("   nm chat start")
    print()
    print("2. First-time login (if prompted):")
    print("   - Enter your phone number (international format: +234...)")
    print("   - Enter the login code Telegram sends you")
    print("   - Enter your 2FA password (if you have one)")
    print()
    print("3. Test it:")
    print("   - Open Telegram on your phone")
    print("   - Send a message to yourself (Saved Messages)")
    print("   - Type: /game")
    print("   - The bot should respond!")
    print()
    print("=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print()
    print("If you encounter issues:")
    print("  - Check logs: tail -f ~/.nomorals/logs/nomorals.log")
    print("  - Read: TELEGRAM_SETUP.md")
    print("  - Ensure telethon is installed: pip install telethon")
    print()

if __name__ == "__main__":
    main()
