"""The ``games`` tool: Devon plays, hosts, and invents games through the spine.

This is how the brain reaches the game engine from plain language —
"let's play something" becomes a real game with real state, not a command
menu. Actions: list | start | move | state | join | spectate | invent |
tournament.

Games are PUBLIC: no owner gating here. Anyone in a group can play —
per the standing rule, games are never gated for outsiders.
"""
from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

_engines: dict[int, Any] = {}


def _engine(registry: Any) -> Any:
    """One GameEngine per registry (shared rooms, shared players)."""
    key = id(registry)
    eng = _engines.get(key)
    if eng is None:
        from ..games.engine import GameEngine
        context = getattr(registry, "context", None)
        suggest = getattr(context, "suggest", None) if context else None
        eng = GameEngine(context, send=lambda chat, text: None,
                         suggest=suggest)
        # invented games join the roster
        try:
            from ..games.invent import load_designs, InventedGame
            for design in load_designs():
                try:
                    eng.register(InventedGame(design, suggest=suggest))
                except Exception:  # noqa: BLE001 - one bad design can't block
                    pass
        except Exception:  # noqa: BLE001
            pass
        # character/brain agent seats play through the real engine
        try:
            from ..characters import CharacterStore, register_with_engine
            register_with_engine(eng, CharacterStore(), suggest=suggest)
        except Exception:  # noqa: BLE001
            pass
        _engines[key] = eng
    return eng


def _player(platform: str, sender: str, name: str) -> Any:
    from ..games.players import Player
    return Player.from_sender(platform, sender, name or sender)


def register(registry: Any) -> None:
    @registry.register(
        "games",
        description=(
            "Play, host, and invent games. Actions: list (what's playable), "
            "start (open a game: game name, players), move (play a turn), "
            "state (current board/transcript), join (enter a live game), "
            "spectate (watch), invent (design a brand-new game from a theme), "
            "tournament (start a multi-game bracket). Games are public — "
            "anyone can play."
        ),
        capability="",
    )
    def games(
        action: str,
        *,
        game: str = "",
        players: str = "",
        move: str = "",
        room: str = "",
        theme: str = "",
        chat: str = "spine",
        name: str = "",
        rounds: int = 3,
        stake: int = 0,
    ) -> dict[str, Any]:
        """Run a game action and return the result."""
        action = (action or "").strip().lower()
        eng = _engine(registry)

        if action == "list":
            items = []
            for gname, g in sorted(eng.games.items()):
                items.append({
                    "name": gname,
                    "title": getattr(g, "title", gname),
                    "blurb": getattr(g, "blurb", ""),
                    "min": getattr(g, "min_players", 1),
                    "max": getattr(g, "max_players", 8),
                    "invented": gname in _invented_names(eng),
                })
            return {"ok": True, "games": items, "count": len(items)}

        if action == "invent":
            if not theme:
                return {"ok": False, "error": "invent needs a theme"}
            from ..games.invent import design_game, save_design, InventedGame
            context = getattr(registry, "context", None)
            suggest = getattr(context, "suggest", None) if context else None
            try:
                design = design_game(theme, suggest,
                                     n_players=max(2, len(players.split(",")) if players else 4))
            except RuntimeError as exc:
                return {"ok": False, "error": str(exc)}
            save_design(design)
            eng.register(InventedGame(design, suggest=suggest))
            return {"ok": True, "design": design,
                    "note": f"invented '{design['title']}' — start it with action=start game={design['name']}"}

        if action == "tournament":
            from ..games.tournaments import new_tournament
            plist = [p.strip() for p in players.split(",") if p.strip()]
            if not plist:
                return {"ok": False, "error": "tournament needs players"}
            glist = [g.strip() for g in game.split(",") if g.strip()] or ["wordchain"]
            # validate games exist
            bad = [g for g in glist if g.lower() not in eng.games]
            if bad:
                return {"ok": False, "error": f"unknown games: {', '.join(bad)}"}
            t = new_tournament(name or "Tournament", glist, plist,
                               rounds=rounds, stake=stake)
            note = t.standings_text()
            if stake:
                problems = t.collect_antes(eng.store)
                if problems:
                    return {"ok": False,
                            "error": "stake failed: " + "; ".join(problems)}
                note += f"\n💰 {stake} coins anted from each player"
            return {"ok": True, "tournament": t.to_dict(), "note": note}

        if action in ("start", "join"):
            if not game:
                return {"ok": False, "error": "start needs a game name"}
            plist = [p.strip() for p in players.split(",") if p.strip()]
            host = _player("spine", plist[0] if plist else "devon",
                           plist[0] if plist else "Devon")
            try:
                room_obj, msgs = eng.start(chat, game.lower(), host, kind="group")
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            # seat extra players (characters / humans)
            joined = []
            for pname in plist[1:]:
                try:
                    p = _player("spine", pname, pname)
                    eng.join(chat, p)
                    joined.append(pname)
                except Exception as exc:  # noqa: BLE001
                    joined.append(f"{pname} (failed: {exc})")
            return {"ok": True, "room_id": room_obj.id, "chat": chat,
                    "messages": msgs, "joined": joined}

        if action == "move":
            if not move:
                return {"ok": False, "error": "move needs move text"}
            target = room or chat
            sender = _player("spine", name or "devon", name or "Devon")
            try:
                msgs = eng.move(target, move, sender)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "messages": msgs}

        if action == "state":
            target = room or chat
            rm = eng._rooms.get(target) or eng._by_id.get(target)
            if rm is None:
                return {"ok": False, "error": "no live game there"}
            return {"ok": True, "game": rm.game, "status": rm.status,
                    "players": [p.name for p in rm.players],
                    "transcript": rm.transcript(limit=20)}

        if action == "spectate":
            target = room or chat
            rm = eng._rooms.get(target) or eng._by_id.get(target)
            if rm is None:
                return {"ok": False, "error": "nothing live to watch"}
            return {"ok": True, "game": rm.game,
                    "watching": f"{rm.game} with " + ", ".join(p.name for p in rm.players),
                    "transcript": rm.transcript(limit=20)}

        return {"ok": False,
                "error": f"unknown action {action!r} — list/start/move/state/join/spectate/invent/tournament"}


def _invented_names(eng: Any) -> set[str]:
    try:
        from ..games.invent import load_designs
        return {d["name"] for d in load_designs()}
    except Exception:  # noqa: BLE001
        return set()
