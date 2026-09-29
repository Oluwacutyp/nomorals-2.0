#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from nomorals.games import GameEngine
from nomorals.games.engine import Player
from nomorals.storage.db import Database

# Simulate a group chat scenario
db = Database(":memory:")
db.migrate()

class MockContext:
    def __init__(self, db):
        self.db = db

context = MockContext(db)
engine = GameEngine(context, send=lambda chat_key, msg: print(f"[{chat_key}] {msg}"))

# Create two players
player1 = Player(key="telegram:123:user1", platform="telegram", name="Alice", is_ai=False)
player2 = Player(key="telegram:123:user2", platform="telegram", name="Bob", is_ai=False)

chat_key = "telegram:123"  # group chat

print("=== Starting connect4 in group chat ===")
try:
    room, msgs = engine.start(chat_key, "connect4", player1, kind="group")
    print(f"✓ Game started: {room.game}")
    print(f"  Room kind: {room.kind}")
    print(f"  Players: {[p.name for p in room.players]}")
    print(f"  Humans: {[p.name for p in room.humans]}")
    print(f"  AI seats: {[p.name for p in room.ai_seats]}")
    print(f"  Min players: {engine.games['connect4'].min_players}")
    print(f"  Max players: {engine.games['connect4'].max_players}")
    print(f"  Needs group: {engine.games['connect4'].needs_group}")
    print(f"  Messages: {msgs[:2]}")
    
    # Try to have player2 join
    print("\n=== Player2 trying to join ===")
    engine.join(chat_key, player2)
    print(f"✓ Player2 joined")
    print(f"  Players now: {[p.name for p in room.players]}")
    
    # Try a move
    print("\n=== Player1 making a move ===")
    engine.move(chat_key, "4", player1)
    print(f"✓ Move made")
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Checking game list ===")
print(f"Total games registered: {len(engine.games)}")
print(f"connect4 in list: {'connect4' in engine.games}")
print(f"Sample games: {list(engine.games.keys())[:10]}")
