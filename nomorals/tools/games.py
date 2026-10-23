"""The ``games`` tool: Devon plays, hosts, and invents games through the spine.

This is how the brain reaches the game engine from plain language —
"let's play something" becomes a real game with real state, not a command
menu. Actions: list | start | move | state | join | spectate | invent |
tournament.

Games are PUBLIC: no owner gating here. Anyone in a group can play —
per the standing rule, games are never gated for outsiders. The one
exception is bank administration: wiping the verified trivia bank is
not a game action, so ``trivia_bank purge`` refuses on this surface
and the purge itself lives in the separate ``trivia_bank_purge``
tool, registered with the owner capability — the same gate the
codebase puts on every owner-only operation.
"""
from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

def _engine(registry: Any) -> Any:
    """One GameEngine per registry (shared rooms, shared players).

    The engine lives ON the registry (``registry._games_engine``),
    not in a module-global dict: a global ``id()``-keyed cache never
    evicted dead registries — their engines (and ticker threads)
    leaked forever, and a new registry reusing an id inherited a
    stale engine bound to a dead context. Bound to the registry, the
    engine dies with it (``GameEngine.__del__`` shuts the ticker down).
    """
    eng = getattr(registry, "_games_engine", None)
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
        try:
            registry._games_engine = eng
        except Exception:  # noqa: BLE001 - exotic registry without attrs
            pass
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
            "tournament (start a multi-game bracket), trivia_bank (the "
            "verified Library question bank behind trivia: sub=status "
            "(stock by source + admission rate) | refill (run one "
            "refiller tick)). Purging the bank is owner-only and is "
            "NOT an action of this tool — it runs through the "
            "owner-gated trivia_bank_purge tool. Games are public — "
            "anyone can play."
        ),
        capability="social.play",
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
        sub: str = "",
    ) -> dict[str, Any]:
        """Run a game action and return the result."""
        action = (action or "").strip().lower()
        eng = _engine(registry)

        if action == "trivia_bank":
            forge = eng.forge
            sub_act = (sub or "status").strip().lower()
            if sub_act == "status":
                return {"ok": True, "bank": forge.bank_status(),
                        "note": forge.render_bank_status()}
            if sub_act == "refill":
                return {"ok": True, "refill": forge.refill_bank(),
                        "bank": forge.bank_status(),
                        "note": forge.render_bank_status()}
            if sub_act == "purge":
                return {"ok": False,
                        "error": "trivia_bank purge is owner-only — "
                                 "refused on the public games surface. "
                                 "Purging the verified bank runs "
                                 "through the owner-gated "
                                 "trivia_bank_purge tool."}
            return {"ok": False,
                    "error": f"unknown trivia_bank sub {sub_act!r} — "
                             "status|refill"}

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

    @registry.register(
        "trivia_bank_purge",
        description=(
            "Purge the verified trivia bank behind the games tool: "
            "drop one banked question by its exact text (question=), "
            "or the whole bank when question is empty. OWNER ONLY — "
            "the bank is verified stock; wiping it is an owner "
            "operation, never a public game action."
        ),
        capability="owner",
    )
    def trivia_bank_purge(*, question: str = "") -> dict[str, Any]:
        """Run the bank purge (owner capability enforced by the
        registry at call time)."""
        forge = _engine(registry).forge
        return {"ok": True, **forge.purge_bank(question)}


def _invented_names(eng: Any) -> set[str]:
    try:
        from ..games.invent import load_designs
        return {d["name"] for d in load_designs()}
    except Exception:  # noqa: BLE001
        return set()
