"""Game relay: connect two separate DM chats for multiplayer.

Allows User A (in DM with bot) to invite User B (in their own DM with bot)
to play a game together. The bot relays moves between the two chats.

Usage:
  User A: /game invite <friend_username> <game_name>
  Bot sends invite to User B's DM
  User B: /game accept <invite_code>
  Bot creates a virtual room connecting both DMs
  Both users play in their own DMs, bot relays moves
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from .engine import GameEngine, Player, Room


@dataclass
class GameInvite:
    """A pending game invitation between two users."""
    code: str
    game_name: str
    from_chat: str  # chat_key of inviter
    from_player: Player
    to_username: str  # username of invitee
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    
    def __post_init__(self):
        if not self.expires_at:
            self.expires_at = self.created_at + 3600  # 1 hour
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


@dataclass
class RelayRoom:
    """A virtual room connecting two separate DM chats."""
    room_id: str
    game_name: str
    chat_a: str  # chat_key of player A
    chat_b: str  # chat_key of player B
    player_a: Player
    player_b: Player
    active: bool = True
    last_activity: float = field(default_factory=time.time)


class GameRelay:
    """Manages game invites and relay rooms between separate DM chats."""
    
    def __init__(self, engine: GameEngine):
        self.engine = engine
        self.invites: dict[str, GameInvite] = {}  # code -> invite
        self.relays: dict[str, RelayRoom] = {}  # room_id -> relay
        self.chat_to_relay: dict[str, str] = {}  # chat_key -> room_id
    
    def create_invite(self, from_chat: str, from_player: Player,
                      to_username: str, game_name: str) -> GameInvite:
        """Create a game invite from one user to another."""
        # Check if game exists
        if game_name not in self.engine.games:
            raise ValueError(f"unknown game {game_name!r}")
        
        # Generate unique code
        code = secrets.token_urlsafe(8)
        while code in self.invites:
            code = secrets.token_urlsafe(8)
        
        invite = GameInvite(
            code=code,
            game_name=game_name,
            from_chat=from_chat,
            from_player=from_player,
            to_username=to_username
        )
        self.invites[code] = invite
        return invite
    
    def accept_invite(self, code: str, to_chat: str, to_player: Player) -> RelayRoom:
        """Accept a game invite and create a relay room."""
        if code not in self.invites:
            raise ValueError("invalid or expired invite code")
        
        invite = self.invites[code]
        if invite.is_expired():
            del self.invites[code]
            raise ValueError("invite has expired")
        
        # Create a virtual chat_key for the relay room
        room_id = f"relay:{invite.code}"
        virtual_chat = f"relay:{invite.code}"
        
        # Start the game in the virtual room
        room, msgs = self.engine.start(
            virtual_chat, invite.game_name, invite.from_player, kind="dm"
        )
        
        # Add the second player
        room.players.append(to_player)
        
        # Create relay mapping
        relay = RelayRoom(
            room_id=room_id,
            game_name=invite.game_name,
            chat_a=invite.from_chat,
            chat_b=to_chat,
            player_a=invite.from_player,
            player_b=to_player
        )
        self.relays[room_id] = relay
        self.chat_to_relay[invite.from_chat] = room_id
        self.chat_to_relay[to_chat] = room_id
        
        # Clean up invite
        del self.invites[code]
        
        return relay
    
    def relay_move(self, from_chat: str, text: str, player: Player) -> list[str]:
        """Relay a game move from one DM to the shared game room."""
        if from_chat not in self.chat_to_relay:
            return []
        
        room_id = self.chat_to_relay[from_chat]
        if room_id not in self.relays:
            return []
        
        relay = self.relays[room_id]
        virtual_chat = f"relay:{relay.room_id}"
        
        # Make the move in the virtual room
        msgs = self.engine.move(virtual_chat, text, player)
        
        # Update last activity
        relay.last_activity = time.time()
        
        return msgs
    
    def get_relay_for_chat(self, chat_key: str) -> RelayRoom | None:
        """Get the active relay room for a chat, if any."""
        if chat_key not in self.chat_to_relay:
            return None
        room_id = self.chat_to_relay[chat_key]
        return self.relays.get(room_id)
    
    def close_relay(self, room_id: str):
        """Close a relay room."""
        if room_id not in self.relays:
            return
        
        relay = self.relays[room_id]
        virtual_chat = f"relay:{room_id}"
        
        # Close the game
        self.engine.quit(virtual_chat)
        
        # Clean up mappings
        self.chat_to_relay.pop(relay.chat_a, None)
        self.chat_to_relay.pop(relay.chat_b, None)
        self.relays.pop(room_id, None)
    
    def cleanup_expired(self):
        """Remove expired invites and inactive relays."""
        now = time.time()
        
        # Clean expired invites
        expired = [code for code, inv in self.invites.items() if inv.is_expired()]
        for code in expired:
            del self.invites[code]
        
        # Clean inactive relays (>24 hours)
        inactive = [rid for rid, relay in self.relays.items()
                    if now - relay.last_activity > 86400]
        for rid in inactive:
            self.close_relay(rid)
