"""``nm music`` — the music player surface: transport, queue, playlists,
history, favorites, plus the song-composition actions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from ..emit import _emit


def _wire_streaming(context: Any) -> None:
    """Attach the Spotify/SoundCloud adapters to the context.

    The ``player`` tool (``nomorals/media/playback.py``, L4) never
    imports the connector layer (L5) — it reads these injected adapters
    off the context instead. This is the L7 wiring point. Idempotent
    per process; failures are reported, never fatal (local files still
    play).
    """
    if getattr(context, "spotify_adapter", None) is not None and \
            getattr(context, "soundcloud_adapter", None) is not None:
        return
    from ...connectors.registry import create_connector
    passphrase = os.environ.get("NM_VAULT_PASSPHRASE", "")
    vault = None
    if passphrase:
        from ...accounts.vault import CredentialVault
        vault = CredentialVault(context.db, master_passphrase=passphrase)
    # SoundCloud is keyless — the connector never reads the vault, so a
    # missing vault (locked) is fine for it.
    try:
        context.soundcloud_adapter = create_connector("soundcloud", vault)
    except Exception as exc:  # noqa: BLE001 - streaming is optional
        print(f"music: soundcloud wiring failed: {exc}", file=sys.stderr)
    if vault is None:
        return  # Spotify's OAuth tokens live in the vault — locked, skip
    try:
        context.spotify_adapter = create_connector("spotify", vault)
    except Exception as exc:  # noqa: BLE001 - streaming is optional
        print(f"music: spotify wiring failed: {exc}", file=sys.stderr)


def _looks_like_file(context: Any, target: str) -> bool:
    """Does ``target`` exist as a local file (absolute or workspace)?"""
    if os.path.isabs(target) and os.path.exists(target):
        return True
    ws = getattr(getattr(context, "settings", None), "workspace_dir", "")
    return bool(ws) and os.path.exists(os.path.join(ws, target))


def _fmt_dur(seconds: float) -> str:
    s = max(0, int(seconds or 0))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_track(t: dict[str, Any], i: int | None = None) -> str:
    name = t.get("title") or t.get("path") or "?"
    bits = f"{i}. {name}" if i is not None else str(name)
    if t.get("artist"):
        bits += f" — {t['artist']}"
    if t.get("album"):
        bits += f" [{t['album']}]"
    if t.get("duration"):
        bits += f" ({_fmt_dur(t['duration'])})"
    return bits


def _fmt_play(v: dict[str, Any]) -> str:
    status = v.get("status", "?")
    cur = v.get("current", "")
    if isinstance(cur, dict):
        cur = cur.get("title", "")
    if status == "no-backend":
        text = f"queued (no audio backend): {cur}\n  {v.get('hint', '')}"
        if v.get("stream_url"):
            # SoundCloud resolved a direct stream: on a backend-less box
            # this URL is the playable artifact
            text += f"\n  stream: {v['stream_url']}"
    elif status in ("error", "unsupported"):
        text = f"{status}: {v.get('error', '?')}"
    else:
        text = f"{status}: {cur}"
    skips = v.get("skipped") or []
    if skips:
        names = ", ".join(str(s.get("title") or s.get("index", "?"))
                          for s in skips)
        text += f"\n  skipped {len(skips)} unplayable: {names}"
    return text


def _cmd_music(args: argparse.Namespace, context: Any) -> int:
    """Compose songs through music_writer; play everything through player /
    music_library."""
    tools = context.tools
    action = getattr(args, "action", "styles") or "styles"
    topic = getattr(args, "topic", "") or ""
    extra = [e for e in (getattr(args, "extra", None) or []) if e]

    def call(tool: str, **kw: Any) -> dict[str, Any] | None:
        out = tools.call(tool, **kw)
        if not out.ok:
            print(f"music: {out.error}", file=sys.stderr)
            return None
        return out.value

    def show(value: dict[str, Any], text: str) -> int:
        _emit(args, value, text)
        return 0

    # ── composition (original surface, unchanged) ──────────────────────────
    if action == "styles":
        out = tools.call("music_writer", action="styles")
        if not out.ok:
            print(f"music: {out.error}", file=sys.stderr)
            return 1
        styles = out.value.get("styles", {})
        return show(out.value,
                    "\n".join(f"  {k:<12} {v.get('label', k)} · {v.get('tempo', '')} "
                              f"bpm · {v.get('mode', '')}" for k, v in styles.items())
                    or "no styles")
    if action == "songs":
        v = call("music_writer", action="song", topic=topic)
        if v is None:
            return 1
        return show(v, json.dumps(v, indent=2, default=str))
    if action == "compose":
        if not topic:
            print("music compose needs a topic — nm music compose \"about what\"",
                  file=sys.stderr)
            return 2
        v = call("music_writer", action="compose", topic=topic,
                 style=getattr(args, "style", "pop") or "pop",
                 title=getattr(args, "title", "") or "",
                 key=getattr(args, "key", "") or "",
                 seed=int(getattr(args, "seed", "0") or 0))
        if v is None:
            return 1
        return show(v,
                    f"composed: {v.get('title', topic)} [{v.get('style', '')}]\n"
                    f"  midi: {v.get('midi_path', '')}\n"
                    f"  melody: {str(v.get('melody_description', ''))[:160]}")

    # ── transport ──────────────────────────────────────────────────────────
    if action in ("play", "add", "pause", "resume", "stop", "next", "prev",
                  "status", "now"):
        # streaming-capable actions: wire the adapters first so Spotify
        # URIs and SoundCloud links queue/play through them
        _wire_streaming(context)
    if action == "play":
        from ...media.playback import PlaybackEngine
        targets = [t for t in [topic, *extra] if t]
        force_spotify = bool(getattr(args, "spotify", False))
        force_soundcloud = bool(getattr(args, "soundcloud", False))
        if not targets:
            v = call("player", action="play")
        elif len(targets) == 1 and targets[0].lstrip("-").isdigit():
            v = call("player", action="play", index=int(targets[0]))
        elif len(targets) == 1:
            t = targets[0]
            src = PlaybackEngine.detect_source(t)
            if force_spotify or src == "spotify":
                v = call("player", action="play_spotify", target=t)
            elif force_soundcloud or src == "soundcloud":
                v = call("player", action="play_soundcloud", target=t)
            elif src == "url":
                v = _play_added(call, args, [t], context)
            elif _looks_like_file(context, t):
                v = _play_added(call, args, [t], context)
            else:
                # bare text: keyless SoundCloud search — works with no
                # accounts wired at all (--spotify forces Spotify search)
                v = call("player", action="play_soundcloud", target=t)
        else:
            v = _play_added(call, args, targets, context)
        return 1 if v is None else show(v, _fmt_play(v))
    if action in ("pause", "resume", "stop", "next", "prev"):
        v = call("player", action=action)
        return 1 if v is None else show(v, _fmt_play(v))
    if action == "status":
        v = call("player", action="status")
        if v is None:
            return 1
        state = "playing" if v.get("playing") else \
            ("paused" if v.get("paused") else "stopped")
        backend = v.get("backend", {})
        bname = backend.get("name", "?") if isinstance(backend, dict) else backend
        return show(v, f"{state} [{bname}] {v.get('current', '')} "
                       f"(#{v.get('position', 0) + 1}/{v.get('queue', 0)}) "
                       f"vol {v.get('volume', 0)}")
    if action == "now":
        v = call("player", action="now")
        if v is None:
            return 1
        t = v.get("track") or {}
        line = f"♪ {_fmt_track(t)}"
        pos, dur = v.get("time_pos"), v.get("duration") or 0
        if pos is not None and dur:
            pct = min(100, int(100 * pos / dur))
            line += f"\n  {_fmt_dur(pos)} / {_fmt_dur(dur)} ({pct}%)"
        elif dur:
            line += f"\n  length {_fmt_dur(dur)}"
        state = "playing" if v.get("playing") else \
            ("paused" if v.get("paused") else "stopped")
        flags = [state]
        flags.append(f"repeat:{v.get('repeat', 'off')}")
        if v.get("shuffle"):
            flags.append("shuffle:on")
        if v.get("liked"):
            flags.append("♥ liked")
        line += (f"\n  {' · '.join(flags)} · vol {v.get('volume', 0)} · "
                 f"#{v.get('position', 0) + 1}/{v.get('queue', 0)}")
        return show(v, line)
    if action == "volume":
        if not topic:
            print("music volume needs a level 0-100", file=sys.stderr)
            return 2
        v = call("player", action="volume", level=int(topic))
        return 1 if v is None else show(v, f"volume {v.get('level')}")
    if action == "seek":
        if not topic:
            print("music seek needs seconds", file=sys.stderr)
            return 2
        v = call("player", action="seek", seconds=float(topic))
        if v is None:
            return 1
        msg = f"seeked to {_fmt_dur(v.get('seconds', float(topic)))}"
        if v.get("status") == "unsupported":
            msg += f" (not supported: {v.get('error', '')})"
        return show(v, msg)

    # ── queue ──────────────────────────────────────────────────────────────
    if action == "queue":
        v = call("player", action="queue")
        if v is None:
            return 1
        items = v.get("queue", [])
        if not items:
            return show(v, "queue is empty")
        return show(v, "\n".join(_fmt_track(t, t.get("index")) for t in items))
    if action == "add":
        targets = [t for t in [topic, *extra] if t]
        if not targets:
            print("music add needs file(s) — nm music add song.mp3 …",
                  file=sys.stderr)
            return 2
        v = call("player", action="add", targets="|".join(targets),
                 title=getattr(args, "title", "") or "")
        if v is None:
            return 1
        lines = [f"added {len(v.get('added', []))} "
                 f"(queue: {v.get('queue', 0)})"]
        lines += [_fmt_track(t) for t in v.get("added", [])]
        return show(v, "\n".join(lines))
    if action == "remove":
        if not topic.lstrip("-").isdigit():
            print("music remove needs an index — nm music remove 2",
                  file=sys.stderr)
            return 2
        v = call("player", action="remove", index=int(topic))
        return 1 if v is None else show(
            v, f"removed {v.get('removed')} (queue: {v.get('queue', 0)})")
    if action == "move":
        to = extra[0] if extra else (getattr(args, "to", "") or "")
        if not (topic.lstrip("-").isdigit() and str(to).lstrip("-").isdigit()):
            print("music move needs two indices — nm music move 2 0",
                  file=sys.stderr)
            return 2
        v = call("player", action="move", index=int(topic), to=int(to))
        return 1 if v is None else show(
            v, f"moved {v.get('moved')} → #{v.get('to')}")
    if action == "clear":
        v = call("player", action="clear")
        return 1 if v is None else show(v, "queue cleared")
    if action == "shuffle":
        v = call("player", action="shuffle",
                 on=(topic or "toggle").strip().lower())
        return 1 if v is None else show(
            v, f"shuffle {'on' if v.get('shuffle') else 'off'}")
    if action == "repeat":
        mode = topic.strip().lower() or \
            (getattr(args, "mode", "") or "").strip().lower()
        v = call("player", action="repeat", mode=mode)
        return 1 if v is None else show(v, f"repeat: {v.get('repeat')}")

    # ── playlists ──────────────────────────────────────────────────────────
    if action == "playlists":
        v = call("music_library", action="playlists")
        if v is None:
            return 1
        pls = v.get("playlists", [])
        if not pls:
            return show(v, "no playlists yet — nm music playlist-create <name>")
        return show(v, "\n".join(
            f"  {p['name']} — {p['tracks']} tracks"
            + (f" ({_fmt_dur(p['seconds'])})" if p.get("seconds") else "")
            for p in pls))
    if action == "playlist":
        if not topic:
            return _cmd_music(_with(args, action="playlists"), context)
        v = call("music_library", action="playlist", name=topic,
                 limit=int(getattr(args, "limit", "20") or 20))
        if v is None:
            return 1
        items = v.get("items", [])
        head = f"{v.get('name')} — {v.get('tracks', 0)} tracks"
        if v.get("description"):
            head += f" · {v['description']}"
        lines = [head] + [_fmt_track(t, t.get("index")) for t in items]
        return show(v, "\n".join(lines))
    if action == "playlist-create":
        if not topic:
            print("music playlist-create needs a name", file=sys.stderr)
            return 2
        v = call("music_library", action="playlist_create", name=topic,
                 description=getattr(args, "desc", "") or "")
        return 1 if v is None else show(v, f"playlist created: {v.get('created')}")
    if action == "playlist-delete":
        if not topic:
            print("music playlist-delete needs a name", file=sys.stderr)
            return 2
        v = call("music_library", action="playlist_delete", name=topic)
        return 1 if v is None else show(
            v, f"deleted playlist {v.get('deleted')} "
               f"({v.get('tracks_removed', 0)} tracks)")
    if action == "playlist-rename":
        new = extra[0] if extra else (getattr(args, "name", "") or "")
        if not (topic and new):
            print("music playlist-rename needs old and new names",
                  file=sys.stderr)
            return 2
        v = call("music_library", action="playlist_rename", name=topic,
                 new_name=new)
        return 1 if v is None else show(
            v, f"renamed {v.get('renamed')} → {v.get('to')}")
    if action == "playlist-add":
        if not topic or not extra:
            print("music playlist-add needs a name and file(s) — "
                  "nm music playlist-add gym a.mp3 b.mp3", file=sys.stderr)
            return 2
        v = call("music_library", action="playlist_add", name=topic,
                 targets="|".join(extra))
        if v is None:
            return 1
        lines = [f"added {len(v.get('added', []))} to {v.get('playlist')} "
                 f"(tracks: {v.get('tracks', 0)})"]
        lines += [_fmt_track(t) for t in v.get("added", [])]
        return show(v, "\n".join(lines))
    if action == "playlist-remove":
        if not topic or not extra or not extra[0].lstrip("-").isdigit():
            print("music playlist-remove needs a name and index — "
                  "nm music playlist-remove gym 2", file=sys.stderr)
            return 2
        v = call("music_library", action="playlist_remove", name=topic,
                 index=int(extra[0]))
        return 1 if v is None else show(
            v, f"removed {v.get('removed')} (tracks: {v.get('tracks', 0)})")
    if action == "playlist-play":
        if not topic:
            print("music playlist-play needs a name", file=sys.stderr)
            return 2
        v = call("player", action="playlist_play", name=topic)
        return 1 if v is None else show(
            v, f"playing playlist {v.get('playlist')} "
               f"({v.get('loaded', 0)} tracks)\n" + _fmt_play(v))
    if action == "playlist-save":
        if not topic:
            print("music playlist-save needs a name — the current queue "
                  "is saved as that playlist", file=sys.stderr)
            return 2
        v = call("player", action="playlist_save", name=topic)
        return 1 if v is None else show(
            v, f"saved {v.get('tracks', 0)} tracks to playlist "
               f"{v.get('saved')}")
    if action == "playlist-export":
        if not topic or not extra:
            print("music playlist-export needs a name and a file — "
                  "nm music playlist-export gym gym.m3u", file=sys.stderr)
            return 2
        v = call("music_library", action="playlist_export", name=topic,
                 path=extra[0])
        return 1 if v is None else show(
            v, f"exported {v.get('tracks', 0)} tracks to {v.get('file')}")
    if action == "playlist-import":
        if not topic or not extra:
            print("music playlist-import needs a name and an M3U file — "
                  "nm music playlist-import gym gym.m3u", file=sys.stderr)
            return 2
        v = call("music_library", action="playlist_import", name=topic,
                 path=extra[0])
        return 1 if v is None else show(
            v, f"imported {v.get('added', 0)} tracks into {v.get('imported')} "
               f"(skipped {v.get('skipped', 0)} stale)")
    if action == "playlist-move":
        src = extra[0] if extra else ""
        dst = extra[1] if len(extra) > 1 else (getattr(args, "to", "") or "")
        if not topic or not (str(src).lstrip("-").isdigit() and
                             str(dst).lstrip("-").isdigit()):
            print("music playlist-move needs a name and two indices — "
                  "nm music playlist-move gym 2 0", file=sys.stderr)
            return 2
        v = call("music_library", action="playlist_move", name=topic,
                 index=int(src), to=int(dst))
        return 1 if v is None else show(
            v, f"moved {v.get('moved')} → #{v.get('to')}")
    if action == "playlist-clear":
        if not topic:
            print("music playlist-clear needs a name", file=sys.stderr)
            return 2
        v = call("music_library", action="playlist_clear", name=topic)
        return 1 if v is None else show(v, f"cleared playlist {v.get('cleared')}")

    # ── history / favorites / search / stats ───────────────────────────────
    if action == "history":
        v = call("music_library", action="history",
                 limit=int(getattr(args, "limit", "20") or 20))
        if v is None:
            return 1
        hist = v.get("history", [])
        if not hist:
            return show(v, "no history yet")
        return show(v, "\n".join(_fmt_track(t) for t in hist))
    if action == "history-clear":
        v = call("music_library", action="history_clear")
        return 1 if v is None else show(
            v, f"history cleared ({v.get('removed', 0)} entries)")
    if action == "top":
        v = call("music_library", action="top",
                 limit=int(getattr(args, "limit", "20") or 20))
        if v is None:
            return 1
        top = v.get("top", [])
        if not top:
            return show(v, "no plays recorded yet")
        return show(v, "\n".join(
            f"{t.get('plays', 0)}× {_fmt_track(t)}" for t in top))
    if action == "like":
        v = call("music_library", action="like", path=topic)
        return 1 if v is None else show(v, f"♥ liked {v.get('title')}")
    if action == "unlike":
        v = call("music_library", action="unlike", path=topic)
        return 1 if v is None else show(v, f"unliked {v.get('title')}")
    if action == "liked":
        v = call("music_library", action="liked",
                 limit=int(getattr(args, "limit", "20") or 20))
        if v is None:
            return 1
        favs = v.get("favorites", [])
        if not favs:
            return show(v, "no favorites yet — nm music like")
        return show(v, "\n".join(_fmt_track(t) for t in favs))
    if action == "search":
        query = " ".join([topic, *extra]).strip()
        if not query:
            print("music search needs a query", file=sys.stderr)
            return 2
        v = call("music_library", action="search", query=query)
        if v is None:
            return 1
        lines: list[str] = []
        if v.get("favorites"):
            lines += ["♥ favorites:"] + \
                     [f"  {_fmt_track(t)}" for t in v["favorites"]]
        if v.get("history"):
            lines += ["recently played:"] + \
                     [f"  {_fmt_track(t)}" for t in v["history"]]
        if v.get("playlists"):
            lines += ["playlists:"] + [f"  {n}" for n in v["playlists"]]
        if v.get("playlist_tracks"):
            lines += ["playlist tracks:"] + \
                     [f"  {_fmt_track(t)} ({t.get('playlist', '')})"
                      for t in v["playlist_tracks"]]
        if v.get("queue"):
            lines += ["queue:"] + [f"  {_fmt_track(t)}" for t in v["queue"]]
        return show(v, "\n".join(lines) or f"no matches for {query!r}")
    if action == "stats":
        v = call("music_library", action="stats")
        if v is None:
            return 1
        return show(v,
                    f"playlists: {v.get('playlists', 0)} "
                    f"({v.get('playlist_tracks', 0)} tracks) · "
                    f"favorites: {v.get('favorites', 0)} "
                    f"({_fmt_dur(v.get('favorites_seconds', 0))}) · "
                    f"plays: {v.get('plays', 0)} "
                    f"({v.get('unique_tracks_played', 0)} unique) · "
                    f"queue: {v.get('queue', 0)}")

    print(f"music: unknown action {action!r}", file=sys.stderr)
    return 2


def _with(args: argparse.Namespace, **kw: Any) -> argparse.Namespace:
    """A shallow copy of args with overrides (for action delegation)."""
    ns = argparse.Namespace(**vars(args))
    for k, val in kw.items():
        setattr(ns, k, val)
    return ns


def _play_added(call: Any, args: argparse.Namespace, targets: list[str],
                context: Any) -> dict[str, Any] | None:
    """Add targets to the queue (files, URLs, Spotify/SoundCloud links —
    the player classifies each) and play the first one added."""
    q = call("player", action="queue") or {}
    n_before = len(q.get("queue", []))
    added = call("player", action="add", targets="|".join(targets),
                 title=getattr(args, "title", "") or "")
    if added is None:
        return None
    return call("player", action="play", index=n_before)
