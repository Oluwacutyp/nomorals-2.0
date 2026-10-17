"""Per-chat reply modes + per-chat feature toggles.

One chat can be set to games-only: Devon sends NO reply to normal chat
messages there, but game commands (and game moves while a game is live)
still work, and owner control commands still work (so the owner can
unset it). The use case is a private chat (e.g. with the owner's
partner) where Devon should stay out of the conversation but the table
stays open.

One chat can also be set to games-off: normal chat works, but game
commands are refused honestly (never silently dropped, never run).
Games-off is a mode, not a toggle — ``/chatmode games-on`` restores
normal — and it composes with the feature toggles below.

One chat can be set to quiet (spec §7, decision D1: quiet mode YES):
Devon never messages first there — no proactive/pulse/scheduled sends
into that chat — but replies when addressed. Quiet is the mode view of
the ``proactive`` toggle (two views, never two stores): setting mode
quiet turns proactive off; turning proactive back on restores normal.

Per-chat feature toggles disable one feature in one chat only:
``games``, ``music``, ``voice``, ``images``, ``proactive``. A disabled
feature never runs in that chat: game commands get an honest refusal,
music/voice/image commands get an honest refusal, outbound voice
replies fall back to text, and proactive sends into the chat are held.
Voice-off is text-replies-only (decision D3): inbound voice notes
STILL transcribe — only the voice reply is suppressed. Disabling in
Group A never affects Group B or DMs.

Per-game toggles (god-tier, decision D5): ``games.<game>`` disables one
game in one chat (e.g. ``games.chess``) while the rest keep running.
They compose under the ``games`` toggle: games-off silences every game
regardless of per-game settings; a per-game off never re-enables a
game the ``games`` toggle silenced.

Storage follows the voice-conversation pattern exactly: a JSON file
``{"chats": {chat_key: {"mode": ..., "set_at": ...,
"disabled": {"games": true, ...}}}}`` with tolerant load/save. The
store lives at ``~/.config/nomorals/chat_modes.json`` (env override
``NOMORALS_CHAT_MODES_PATH`` for tests). Unknown chats default to
``normal`` with every feature on.

Backward compatibility: entries written before toggles existed have
no ``"disabled"`` key — they load as all-features-on. Entries with
mode ``normal`` and no disabled feature are pruned (normal is the
default), exactly as before.

Consistency rules (a chat can never be in a contradictory state):
- ``games_only`` implies games ON: setting mode games-only clears a
  disabled-games toggle AND any per-game offs, and disabling games
  while games-only switches the mode back to normal (otherwise the
  chat would be silent for everything — a trap).
- ``games_off`` implies games OFF: enabling games (``games-on`` or
  ``feature games on``) while games-off switches the mode to normal.
- ``quiet`` implies proactive OFF: setting mode quiet turns proactive
  off; turning proactive on while quiet switches the mode to normal.
  Setting mode normal keeps the proactive toggle as-is (so normal +
  proactive-off behaves exactly like quiet, just represented as a
  toggle instead of a mode).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

_log = logging.getLogger("nomorals.partner.chat_modes")


MODE_NORMAL = "normal"
MODE_GAMES_ONLY = "games_only"
MODE_GAMES_OFF = "games_off"
MODE_QUIET = "quiet"
MODES = (MODE_NORMAL, MODE_GAMES_ONLY, MODE_GAMES_OFF, MODE_QUIET)

# Features that can be toggled per chat. Keep this small and stable:
# every entry here needs a gate in the runtime (see FEATURE_COMMANDS
# in nomorals/agents/partner/runtime.py) or the toggle would be a lie.
# ``proactive`` is gated at the notifier (the one proactive funnel),
# not at a command — it has no commands.
FEATURES = ("games", "music", "voice", "images", "proactive")

def _store_path(path: str = "") -> str:
    if path:
        return os.path.expanduser(path)
    override = os.environ.get("NOMORALS_CHAT_MODES_PATH", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".config",
                        "nomorals", "chat_modes.json")


def _load(path: str = "") -> dict[str, Any]:
    try:
        with open(_store_path(path), encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(state: dict[str, Any], path: str = "") -> None:
    try:
        target = _store_path(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, target)
    except OSError as exc:
        _log.warning("chat modes save failed: %s", exc)


def _entry(chats: dict[str, Any], chat_key: str) -> dict[str, Any]:
    entry = chats.get(chat_key)
    if not isinstance(entry, dict):
        return {}
    return entry


def _disabled_of(entry: dict[str, Any]) -> dict[str, bool]:
    """The entry's disabled-feature map, normalised to {feature: True}."""
    disabled = entry.get("disabled")
    if not isinstance(disabled, dict):
        return {}
    return {f: True for f, v in disabled.items()
            if f in FEATURES and v}


def _games_disabled_of(entry: dict[str, Any]) -> dict[str, bool]:
    """The entry's per-game off map, normalised to {game: True}."""
    off = entry.get("games_disabled")
    if not isinstance(off, dict):
        return {}
    return {str(g).strip().lower(): True for g, v in off.items()
            if str(g).strip() and v}


def _store_entry(chats: dict[str, Any], chat_key: str, entry: dict[str, Any]) -> None:
    """Write an entry back, pruning it when it is fully default.

    An entry is default when mode is normal AND no feature is disabled
    AND no game is disabled — keeping it would grow the file forever
    with no-ops.
    """
    mode = entry.get("mode", MODE_NORMAL)
    if (mode == MODE_NORMAL and not _disabled_of(entry)
            and not _games_disabled_of(entry)):
        chats.pop(chat_key, None)
        return
    out: dict[str, Any] = {"mode": mode if mode in MODES else MODE_NORMAL,
                           "set_at": entry.get("set_at") or time.time()}
    disabled = _disabled_of(entry)
    if disabled:
        out["disabled"] = disabled
    games_disabled = _games_disabled_of(entry)
    if games_disabled:
        out["games_disabled"] = games_disabled
    chats[chat_key] = out


def set_chat_mode(chat_key: str, mode: str, path: str = "") -> str:
    """Set a chat's reply mode. Returns the stored mode.

    Raises :class:`ValueError` for an unknown mode or an empty chat key.
    Setting ``normal`` removes the entry when no feature is disabled
    (normal is the default); disabled features are preserved.
    Setting ``games_only`` clears a disabled-games toggle AND any
    per-game offs (games-only implies games on); setting ``games_off``
    keeps the toggles as-is. Setting ``quiet`` turns proactive off;
    setting ``normal`` keeps the proactive toggle as-is.
    """
    mode = (mode or "").strip().lower()
    if mode not in MODES:
        raise ValueError(f"unknown chat mode {mode!r} — want one of: {', '.join(MODES)}")
    chat_key = (chat_key or "").strip()
    if not chat_key:
        raise ValueError("chat key is required")
    state = _load(path)
    chats = state.get("chats", {})
    if not isinstance(chats, dict):
        chats = {}
    entry = _entry(chats, chat_key)
    entry["mode"] = mode
    if mode == MODE_GAMES_ONLY:
        # games-only implies games ON — a disabled-games toggle here
        # would silence the chat for everything.
        disabled = _disabled_of(entry)
        disabled.pop("games", None)
        entry["disabled"] = disabled
        entry["games_disabled"] = {}
    if mode == MODE_QUIET:
        # Quiet IS the proactive-off mode: persist the toggle too, so
        # quiet -> normal keeps proactive off (toggle view of the same
        # setting), instead of silently losing it to pruning.
        disabled = _disabled_of(entry)
        disabled["proactive"] = True
        entry["disabled"] = disabled
    _store_entry(chats, chat_key, entry)
    state["chats"] = chats
    _save(state, path)
    return mode


def get_chat_mode(chat_key: str, path: str = "") -> str:
    """A chat's reply mode; ``normal`` when never set."""
    chats = _load(path).get("chats", {})
    if not isinstance(chats, dict):
        return MODE_NORMAL
    entry = chats.get((chat_key or "").strip(), {})
    mode = entry.get("mode", MODE_NORMAL) if isinstance(entry, dict) else MODE_NORMAL
    return mode if mode in MODES else MODE_NORMAL


def is_games_only(chat_key: str, path: str = "") -> bool:
    return get_chat_mode(chat_key, path) == MODE_GAMES_ONLY


def is_games_off(chat_key: str, path: str = "") -> bool:
    return get_chat_mode(chat_key, path) == MODE_GAMES_OFF


def is_quiet(chat_key: str, path: str = "") -> bool:
    return get_chat_mode(chat_key, path) == MODE_QUIET


def set_feature(chat_key: str, feature: str, enabled: bool,
                path: str = "") -> bool:
    """Enable/disable one feature in one chat. Returns the stored state.

    ``enabled=False`` disables the feature. Raises :class:`ValueError`
    for an unknown feature or an empty chat key.
    Disabling games while games-only drops the mode to normal;
    enabling games while games-off drops the mode to normal.
    Disabling proactive while normal sets the mode to quiet (quiet IS
    the proactive-off mode); enabling proactive while quiet restores
    normal. Dotted names (``games.chess``) are per-game toggles —
    use :func:`set_game` for those.
    """
    feature = (feature or "").strip().lower()
    if "." in feature:
        raise ValueError(
            f"{feature!r} is a per-game toggle — use /chatmode feature "
            f"games.<game> on|off (handled by set_game)")
    if feature not in FEATURES:
        raise ValueError(f"unknown feature {feature!r} — want one of: {', '.join(FEATURES)}")
    chat_key = (chat_key or "").strip()
    if not chat_key:
        raise ValueError("chat key is required")
    state = _load(path)
    chats = state.get("chats", {})
    if not isinstance(chats, dict):
        chats = {}
    entry = _entry(chats, chat_key)
    mode = entry.get("mode", MODE_NORMAL) if isinstance(entry, dict) else MODE_NORMAL
    if mode not in MODES:
        mode = MODE_NORMAL
    entry["mode"] = mode
    disabled = _disabled_of(entry)
    if enabled:
        disabled.pop(feature, None)
        if feature == "games" and mode == MODE_GAMES_OFF:
            # games-on while games-off: the mode IS the games switch.
            entry["mode"] = MODE_NORMAL
        if feature == "proactive" and mode == MODE_QUIET:
            # proactive-on while quiet: quiet IS the proactive-off mode.
            entry["mode"] = MODE_NORMAL
    else:
        disabled[feature] = True
        if feature == "games" and mode == MODE_GAMES_ONLY:
            # games-only with games off would silence everything.
            entry["mode"] = MODE_NORMAL
        if feature == "proactive" and mode == MODE_NORMAL:
            # proactive-off while normal: the mode view is quiet.
            entry["mode"] = MODE_QUIET
    entry["disabled"] = disabled
    _store_entry(chats, chat_key, entry)
    state["chats"] = chats
    _save(state, path)
    return enabled


def set_game(chat_key: str, game: str, enabled: bool, path: str = "") -> bool:
    """Enable/disable ONE game in one chat (per-game toggle, D5).

    ``enabled=False`` turns just that game off; every other game keeps
    running. Raises :class:`ValueError` for an empty game name or chat
    key. Setting a per-game off while games-only drops the mode to
    normal (games-only implies ALL games on); turning a game back on
    never re-enables what the ``games`` toggle silenced.
    """
    game = (game or "").strip().lower()
    if not game or "." in game:
        raise ValueError(f"bad game name {game!r} — want a single game name")
    chat_key = (chat_key or "").strip()
    if not chat_key:
        raise ValueError("chat key is required")
    state = _load(path)
    chats = state.get("chats", {})
    if not isinstance(chats, dict):
        chats = {}
    entry = _entry(chats, chat_key)
    mode = entry.get("mode", MODE_NORMAL) if isinstance(entry, dict) else MODE_NORMAL
    if mode not in MODES:
        mode = MODE_NORMAL
    off = _games_disabled_of(entry)
    if enabled:
        off.pop(game, None)
    else:
        off[game] = True
        if mode == MODE_GAMES_ONLY:
            # games-only implies ALL games on.
            entry["mode"] = MODE_NORMAL
    entry["games_disabled"] = off
    _store_entry(chats, chat_key, entry)
    state["chats"] = chats
    _save(state, path)
    return enabled


def is_game_disabled(chat_key: str, game: str, path: str = "") -> bool:
    """Is this one game off in this chat?

    True when the whole ``games`` feature is off (games-off mode or
    games toggle) OR this game is per-game off. A per-game off never
    re-enables a game the ``games`` toggle silenced.
    """
    if is_feature_disabled(chat_key, "games", path):
        return True
    game = (game or "").strip().lower()
    if not game:
        return False
    chats = _load(path).get("chats", {})
    if not isinstance(chats, dict):
        return False
    entry = chats.get((chat_key or "").strip(), {})
    if not isinstance(entry, dict):
        return False
    return bool(_games_disabled_of(entry).get(game, False))


def game_disabled_reply(game: str) -> str:
    """The honest refusal for a per-game-off game."""
    game = (game or "").strip()
    return (f"{game or 'That game'} is turned off in this chat — it doesn't "
            f"run here, but your other games still do. The owner can turn "
            f"it back on with /chatmode feature games.{game or '<game>'} on.")


def proactive_allowed(chat_key: str, path: str = "") -> bool:
    """May proactive/pulse/scheduled sends go into this chat?

    False in quiet mode and when the proactive toggle is off. The
    single check every proactive funnel consults.
    """
    return not is_feature_disabled(chat_key, "proactive", path)


def is_feature_disabled(chat_key: str, feature: str, path: str = "") -> bool:
    """Is this feature disabled in this chat? Unknown features: False."""
    feature = (feature or "").strip().lower()
    if feature not in FEATURES:
        return False
    chats = _load(path).get("chats", {})
    if not isinstance(chats, dict):
        return False
    entry = chats.get((chat_key or "").strip(), {})
    if not isinstance(entry, dict):
        return False
    if feature == "games" and entry.get("mode") == MODE_GAMES_OFF:
        return True
    if feature == "proactive" and entry.get("mode") == MODE_QUIET:
        return True
    return bool(_disabled_of(entry).get(feature, False))


def is_games_disabled(chat_key: str, path: str = "") -> bool:
    """Games off in this chat — via games-off mode or the games toggle."""
    return is_feature_disabled(chat_key, "games", path)


def get_chat_settings(chat_key: str, path: str = "") -> dict[str, Any]:
    """A chat's full per-chat settings: mode + disabled-feature list
    + per-game off list."""
    chats = _load(path).get("chats", {})
    entry: dict[str, Any] = {}
    if isinstance(chats, dict):
        raw = chats.get((chat_key or "").strip(), {})
        if isinstance(raw, dict):
            entry = raw
    mode = entry.get("mode", MODE_NORMAL)
    if mode not in MODES:
        mode = MODE_NORMAL
    disabled = sorted(_disabled_of(entry))
    # games-off mode IS a games disable — report it as such so status
    # never shows "games off" mode with "games on" toggles.
    if mode == MODE_GAMES_OFF and "games" not in disabled:
        disabled = sorted(disabled + ["games"])
    # quiet mode IS a proactive disable — same reporting rule.
    if mode == MODE_QUIET and "proactive" not in disabled:
        disabled = sorted(disabled + ["proactive"])
    return {"mode": mode, "disabled": disabled,
            "games_disabled": sorted(_games_disabled_of(entry))}


# ── command → feature mapping (the gate's single source of truth) ──────
# Which slash-command kinds belong to each toggleable feature. The
# runtime's shared gate (handle_control + the _process game branch)
# consults this, so a toggle can never be bypassed by reaching the
# command through another path (public bot, stack, direct dispatch).
#
# Deliberate exclusions, documented so they don't look like oversights:
# - "arena" is NOT games: it is the content arena (research/stream/
#   review/approve), not the game engine.
# - "say" is NOT voice: it is dual-use (send-as-her + speak-as-voice);
#   gating it would block the send use. /speak is the pure voice path.
# - "dj" maps to music (a music show), even though it has voice breaks.

_MUSIC_COMMANDS = frozenset({"music", "play", "dj"})
_VOICE_COMMANDS = frozenset({
    "tts", "stt", "voice", "vnote", "speak", "voices", "clonevoice",
    "sing", "audio",
})
_IMAGE_COMMANDS = frozenset({"image", "look", "lens", "caption"})


def feature_for_command(kind: str) -> str | None:
    """Which toggleable feature a command kind belongs to, or None.

    ``None`` means the command is not feature-gated (normal chat,
    owner controls, /chatmode itself, …).
    """
    kind = (kind or "").strip().lower()
    if not kind:
        return None
    if kind == "game":
        return "games"
    if kind in _MUSIC_COMMANDS:
        return "music"
    if kind in _VOICE_COMMANDS:
        return "voice"
    if kind in _IMAGE_COMMANDS:
        return "images"
    # Game kinds come from the control registry (never a pasted list),
    # so a new game command is gated the day it is registered.
    from ..social.chat.control import GAME_COMMANDS

    if kind in GAME_COMMANDS:
        return "games"
    return None


def feature_disabled_reply(feature: str) -> str:
    """The honest refusal for a disabled feature — never a silent drop."""
    feature = (feature or "").strip().lower()
    if feature == "games":
        return ("Games are turned off in this chat — game commands don't "
                "run here. The owner can turn them back on with "
                "/chatmode games-on.")
    if feature == "music":
        return ("Music is turned off in this chat — music commands don't "
                "run here. The owner can turn it back on with "
                "/chatmode feature music on.")
    if feature == "voice":
        return ("Voice replies are turned off in this chat — replies "
                "stay text, but voice notes still transcribe. The owner "
                "can turn voice replies back on with "
                "/chatmode feature voice on.")
    if feature == "images":
        return ("Images are turned off in this chat — image commands "
                "don't run here. The owner can turn them back on with "
                "/chatmode feature images on.")
    if feature == "proactive":
        return ("Proactive messages are turned off in this chat — I "
                "won't message first here, but I still reply when "
                "addressed. The owner can turn them back on with "
                "/chatmode feature proactive on.")
    return f"{feature} is turned off in this chat."


def list_chat_modes(path: str = "") -> dict[str, str]:
    """Every chat with a non-default mode, as {chat_key: mode}."""
    chats = _load(path).get("chats", {})
    if not isinstance(chats, dict):
        return {}
    out: dict[str, str] = {}
    for key, entry in chats.items():
        if isinstance(entry, dict) and entry.get("mode") in MODES:
            if entry["mode"] != MODE_NORMAL:
                out[key] = entry["mode"]
    return out


def list_chat_settings(path: str = "") -> dict[str, dict[str, Any]]:
    """Every chat with non-default settings, as {chat_key: settings}.

    A chat appears here when its mode is not normal OR any feature is
    disabled OR any game is disabled — i.e. everything ``/chatmode
    list`` shows.
    """
    chats = _load(path).get("chats", {})
    if not isinstance(chats, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key in chats:
        settings = get_chat_settings(key, path)
        if (settings["mode"] != MODE_NORMAL or settings["disabled"]
                or settings.get("games_disabled")):
            out[key] = settings
    return out
