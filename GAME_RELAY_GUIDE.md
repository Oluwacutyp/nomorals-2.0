# Game Relay System - Play with Friends in DMs

## Overview

The game relay system allows you to play multiplayer games with friends in **separate 1-on-1 DM chats** with the bot, without needing a group chat.

This solves the Telegram limitation where bots can't be added to direct messages between two humans.

## How It Works

1. **You** chat with the bot in your DM
2. **Your friend** chats with the bot in their DM
3. The bot **relays** game moves between your two separate chats
4. You both play the same game, seeing each other's moves

## Usage

### Starting a Game with a Friend

**Step 1: Send an invite**

In your DM with the bot:
```
/game invite @friend_username connect4
```

The bot will send your friend a DM with an invite code.

**Step 2: Friend accepts**

Your friend receives:
```
🎮 Alice invited you to play connect4!
To accept: /game accept abc123
(expires in 1 hour)
```

Your friend replies:
```
/game accept abc123
```

**Step 3: Play!**

Both of you get a message:
```
🎮 Game started! Play here in this chat. Your moves will be relayed to your opponent.
```

Now just type your moves (e.g., `4` for column 4 in Connect Four) and the bot will relay them to your opponent.

## Commands

### `/game invite <username> <game>`
Send a game invite to a friend.

**Example:**
```
/game invite @alice tictactoe
/game invite @bob battleship
```

**Notes:**
- Username should be the Telegram username (with @)
- Invite expires after 1 hour
- You can have multiple pending invites

### `/game accept <code>`
Accept a game invite using the code from the invite message.

**Example:**
```
/game accept abc123
```

**Notes:**
- Code is in the invite message you received
- Once accepted, the game starts immediately
- Both players play in their own DMs

## Available Games

All 32 games in the catalog support relay play:

**Easy games:** wordchain, hangman, numberguess, two_truths, wyrr, spy, auction, trivia

**Medium games:** mafia, king, story, rpg, shop, duel, case

**Ambitious games:** world, escape, political, arena

**Wild games (W95):** poker, ttt, bulls, craps, memory, mines, wordle

**Arcade games (W97):** 2048, snake, connect4, battleship

**Casino games (W98):** blackjack, roulette, slots

## Example: Playing Connect Four

**Alice's DM with bot:**
```
Alice: /game invite @bob connect4
Bot: invite sent to @bob for connect4. They'll get a DM with code xyz789.

[... Bob accepts ...]

Bot: 🎮 Bob accepted! Game started in this chat.
     1 2 3 4 5 6 7
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     Alice (🔴) vs Bob (🟡)
     Your turn — drop a disc (1-7).

Alice: 4
Bot: You dropped in column 4.
     1 2 3 4 5 6 7
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · 🔴 · · ·
     Bob's turn...

[... Bob plays in his DM, you see his move ...]

Bot: Bob dropped in column 3.
     1 2 3 4 5 6 7
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · 🟡 🔴 · · ·
     Your turn...
```

**Bob's DM with bot:**
```
Bob receives:
🎮 Alice invited you to play connect4!
To accept: /game accept xyz789
(expires in 1 hour)

Bob: /game accept xyz789
Bot: 🎮 Game started! Play here in this chat. Your moves will be relayed to your opponent.
     1 2 3 4 5 6 7
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · 🔴 · · ·
     Alice's turn...

[... Alice plays, you see her move ...]

Bot: Alice dropped in column 4.
     1 2 3 4 5 6 7
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · 🔴 · · ·
     Your turn — drop a disc (1-7).

Bob: 3
Bot: You dropped in column 3.
     1 2 3 4 5 6 7
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · · · · · ·
     · · 🟡 🔴 · · ·
     Alice's turn...
```

## Technical Details

### How Relay Works

1. **Invite creation:** Bot generates a unique code and stores the invite
2. **Invite acceptance:** Bot creates a virtual "relay room" connecting both DMs
3. **Move relay:** When you type a move, bot:
   - Processes it in the virtual room
   - Sends the result to BOTH DMs
   - Both players see the same game state

### Limitations

- **Telegram only:** Currently only works with Telegram (other platforms coming soon)
- **Username resolution:** Bot needs to know your friend's Telegram chat ID (may require them to message the bot first)
- **Invite expiry:** Invites expire after 1 hour
- **Room cleanup:** Relay rooms auto-close after 24 hours of inactivity

### Future Improvements

- [ ] Support for Discord, WhatsApp, etc.
- [ ] Better username resolution (search by display name)
- [ ] Spectator mode (watch friends play)
- [ ] Tournament brackets
- [ ] Persistent game history

## Troubleshooting

**Q: My friend didn't receive the invite**
- Make sure they've messaged the bot at least once (bots can't DM users who haven't interacted with them)
- Check the username is correct (with @ symbol)

**Q: The invite code doesn't work**
- Codes expire after 1 hour
- Make sure you're typing it exactly as shown
- Ask your friend to send a new invite

**Q: I don't see my friend's moves**
- Check that both of you are in separate DMs with the bot (not a group)
- Try sending `/game status` to see the current game state

## Comparison: Relay vs Group Chat

| Feature | Relay (DMs) | Group Chat |
|---------|-------------|------------|
| Setup | Easy (just invite) | Need to create group |
| Privacy | 1-on-1 with bot | All members see chat |
| Spectators | No | Yes |
| Platform support | Telegram only | All platforms |
| Max players | 2 | Varies by game |

**Use relay when:**
- You want private 1-on-1 gameplay
- You don't want to create a group
- You're on Telegram

**Use group chat when:**
- You want spectators
- You're on Discord/WhatsApp/etc.
- You need more than 2 players

## Code Reference

### GameRelay Class
Located in `nomorals/games/relay.py`

**Methods:**
- `create_invite(from_chat, from_player, to_username, game_name)` - Create an invite
- `accept_invite(code, to_chat, to_player)` - Accept an invite
- `relay_move(from_chat, text, player)` - Relay a move between DMs
- `get_relay_for_chat(chat_key)` - Get the relay room for a chat
- `close_relay(room_id)` - Close a relay room
- `cleanup_expired()` - Remove expired invites and inactive rooms

### Integration Points
- `partner_runtime.py` - Handles `/game invite` and `/game accept` commands
- `_route_game_move()` - Checks for relay rooms before routing to regular engine
- `_game_relay()` - Lazy-loads the GameRelay instance

## Testing

Run the relay tests:
```bash
cd /home/user/No-morals-ai
python3 -m unittest tests.test_relay -v
```

Expected: All tests pass, demonstrating invite creation, acceptance, and move relay.
