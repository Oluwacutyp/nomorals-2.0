"""``/mix`` — the shareable-mix chat surface.

Thin chat front-end over :mod:`nomorals.media.mix_surface`: one
current mix per chat, kept as a descriptor JSON under the mixes
workdir, edited with the organ's own edit helpers, rendered with
``render_mix`` and delivered with ``deliver`` through an injected
sender. Never raises: every failure is an honest plain-words reply
naming the missing thing.

Subcommands (see :func:`mix_surface.mix_menu`):
``new <title>`` (from the durable queue), ``add <track>`` (queue
index, queue title, or an existing audio file), ``order <n> <m>``,
``curve <late_peak|linear|wave|flat>``, ``card``, ``render``,
``send <platform:chat_id> [--voice]``, ``share``, ``load <json>``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)

CURVES = ("late_peak", "linear", "wave", "flat")


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "").strip().lower()).strip("-")
    return slug or "mix"


def _safe_key(chat_key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", (chat_key or "").strip())
    return safe or "default"


def _workdir(workdir: str | None) -> Path:
    if workdir:
        return Path(workdir)
    return Path.home() / ".devon" / "mixes"


def _state_path(workdir: str | None, chat_key: str) -> Path:
    return _workdir(workdir) / f"{_safe_key(chat_key)}.json"


def _chat_dir(workdir: str | None, chat_key: str) -> Path:
    return _workdir(workdir) / _safe_key(chat_key)


def _load_current(workdir: str | None, chat_key: str) -> Any | None:
    """The chat's current descriptor, or None. Corrupt state is
    reported, never silently dropped."""
    from .mix_surface import load_descriptor

    path = _state_path(workdir, chat_key)
    if not path.exists():
        return None
    try:
        return load_descriptor(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record_failure("mix_chat.load_current", exc)
        return None


def _save_current(workdir: str | None, chat_key: str, descriptor: Any) -> None:
    from .mix_surface import share_descriptor

    path = _state_path(workdir, chat_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(share_descriptor(descriptor), encoding="utf-8")


def _queue_rows(context: Any) -> list[dict[str, Any]]:
    """Durable queue rows (position order), or [] when there is no
    database to read. Never raises."""
    db = getattr(context, "db", None) if context is not None else None
    if db is None:
        return []
    try:
        rows = db.query_rows(
            "SELECT path, title, artist, album, duration, kind "
            "FROM media_queue ORDER BY position, added_at"
        )
    except Exception as exc:  # noqa: BLE001 - no queue is a reply, not a crash
        record_failure("mix_chat.queue_rows", exc)
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            out.append(
                {
                    "path": str(row["path"] or ""),
                    "title": str(row["title"] or ""),
                    "artist": str(row["artist"] or ""),
                    "duration": float(row["duration"] or 0),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _resolve_track(arg: str, rows: list[dict[str, Any]]) -> tuple[Any | None, str]:
    """Resolve an ``add`` argument to a MixTrack, or (None, reason)."""
    from .mix_surface import MixTrack

    arg = (arg or "").strip()
    if not arg:
        return None, "usage: /mix add <track> — a queue number, a queue title, or a file path"
    if arg.isdigit():
        idx = int(arg) - 1
        if 0 <= idx < len(rows):
            row = rows[idx]
            return MixTrack(
                path=row["path"], title=row["title"] or Path(row["path"]).stem,
                duration_s=row["duration"],
            ), ""
        return None, f"no_such_track: queue has {len(rows)} track(s), no #{arg}"
    low = arg.lower()
    matches = [r for r in rows if low in (r["title"] or "").lower()]
    if len(matches) == 1:
        row = matches[0]
        return MixTrack(
            path=row["path"], title=row["title"] or Path(row["path"]).stem,
            duration_s=row["duration"],
        ), ""
    if len(matches) > 1:
        names = "; ".join(f"#{i + 1} {r['title']}" for i, r in enumerate(rows)
                          if low in (r["title"] or "").lower())
        return None, f"ambiguous track '{arg}' — matches: {names} — use a queue number"
    path = Path(arg).expanduser()
    if path.is_file():
        return MixTrack(path=str(path), title=path.stem), ""
    return None, (
        f"no_such_track: '{arg}' is not in the queue and not a file — "
        "add tracks with /play first, or pass a real audio file path"
    )


def _render_reply(descriptor: Any, rendered: Any) -> str:
    from .mix_surface import format_mix_card

    lines = [format_mix_card(descriptor)]
    if rendered.audio_path:
        lines.append(f"\nrendered: {rendered.audio_path} "
                     f"({rendered.total_duration_s:.0f}s)")
    else:
        missing = ", ".join(rendered.missing_tracks) or "no tracks"
        lines.append(f"\nplan only — no audio: missing {missing}. "
                     "Add files that exist on disk, then /mix render again.")
    return "\n".join(lines)


def control_mix(
    tail: str,
    *,
    chat_key: str = "",
    sender: Any = None,
    context: Any = None,
    workdir: str | None = None,
) -> str:
    """``/mix`` chat handler. Never raises."""
    try:
        return _control_mix_inner(
            tail, chat_key=chat_key, sender=sender,
            context=context, workdir=workdir,
        )
    except Exception as exc:  # noqa: BLE001 - chat surface never raises
        record_failure("mix_chat.control_mix", exc)
        return f"mix failed: {exc}"


def _control_mix_inner(
    tail: str,
    *,
    chat_key: str = "",
    sender: Any = None,
    context: Any = None,
    workdir: str | None = None,
) -> str:
    from .mix_surface import (
        MixDescriptor,
        MixTrack,
        deliver,
        format_mix_card,
        load_descriptor,
        mix_menu,
        render_mix,
        share_descriptor,
        with_track_added,
        with_tracks_reordered,
    )

    tail = (tail or "").strip()
    current = _load_current(workdir, chat_key)
    if current is None and _state_path(workdir, chat_key).exists():
        return ("the saved mix for this chat is corrupt — start a new "
                "one with /mix new <title>.")
    if not tail:
        menu = mix_menu()
        if current is None:
            return menu
        return menu + "\n\ncurrent mix:\n" + format_mix_card(current)

    parts = tail.split(None, 1)
    verb = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if verb == "new":
        title = rest or "Untitled mix"
        rows = _queue_rows(context)
        if not rows:
            return ("the queue is empty — add tracks with /play first, "
                    "then /mix new <title>.")
        descriptor = MixDescriptor(
            id=_slug(title), title=title,
            tracks=[
                MixTrack(path=r["path"],
                         title=r["title"] or Path(r["path"]).stem,
                         duration_s=r["duration"])
                for r in rows
            ],
        )
        replaced = ""
        if current is not None:
            replaced = (f" (replaced the previous mix "
                        f"'{current.title}' v{current.version})")
        _save_current(workdir, chat_key, descriptor)
        return (f"🎧 {descriptor.title} — {len(descriptor.tracks)} tracks "
                f"from the queue{replaced}\n" + format_mix_card(descriptor))

    if verb == "load":
        if not rest:
            return "usage: /mix load <descriptor JSON> — paste a /mix share payload"
        try:
            loaded = load_descriptor(rest)
        except (ValueError, TypeError) as exc:
            return f"not a mix descriptor: {exc}"
        _save_current(workdir, chat_key, loaded)
        return ("loaded mix:\n" + format_mix_card(loaded))

    if current is None:
        return "no mix yet — start one with /mix new <title>."

    if verb == "add":
        track, why = _resolve_track(rest, _queue_rows(context))
        if track is None:
            return why
        updated = with_track_added(current, track)
        _save_current(workdir, chat_key, updated)
        return (f"added #{len(updated.tracks)} — {track.title or '(untitled)'}\n"
                + format_mix_card(updated))

    if verb == "order":
        nums = rest.split()
        if len(nums) != 2 or not all(n.isdigit() for n in nums):
            return "usage: /mix order <n> <m> — move track n to position m"
        n, m = int(nums[0]), int(nums[1])
        total = len(current.tracks)
        if not (1 <= n <= total and 1 <= m <= total):
            return f"order out of range: this mix has {total} track(s)"
        order = list(range(total))
        order.insert(m - 1, order.pop(n - 1))
        updated = with_tracks_reordered(current, order)
        _save_current(workdir, chat_key, updated)
        return (f"moved track {n} to position {m}\n"
                + format_mix_card(updated))

    if verb == "curve":
        name = rest.lower()
        if name not in CURVES:
            return ("unknown curve '" + rest + "' — use one of: "
                    + ", ".join(CURVES))
        updated = MixDescriptor.from_dict(current.to_dict())
        updated.energy_curve = name
        updated.version = current.version + 1
        _save_current(workdir, chat_key, updated)
        return (f"energy curve: {name}\n" + format_mix_card(updated))

    if verb == "card":
        return format_mix_card(current)

    if verb == "render":
        chat_dir = _chat_dir(workdir, chat_key)
        chat_dir.mkdir(parents=True, exist_ok=True)
        rendered = render_mix(
            current,
            out_path=chat_dir / f"{current.id or 'mix'}.wav",
            checkpoint_path=chat_dir / f"{current.id or 'mix'}.checkpoint.json",
            resume=True,
        )
        return _render_reply(current, rendered)

    if verb == "send":
        tokens = rest.split()
        if not tokens:
            return "usage: /mix send <platform:chat_id> [--voice]"
        target = tokens[0]
        as_voice = "--voice" in tokens[1:]
        if ":" not in target or not target.split(":", 1)[1]:
            return ("bad chat: use 'platform:chat_id' — e.g. "
                    "/mix send telegram:123456789")
        if sender is None:
            return "can't send from here — no sender is attached to this chat."
        chat_dir = _chat_dir(workdir, chat_key)
        chat_dir.mkdir(parents=True, exist_ok=True)
        rendered = render_mix(
            current,
            out_path=chat_dir / f"{current.id or 'mix'}.wav",
            checkpoint_path=chat_dir / f"{current.id or 'mix'}.checkpoint.json",
            resume=True,
        )
        if rendered.audio_path is None:
            missing = ", ".join(rendered.missing_tracks) or "no tracks"
            return (f"nothing to send — plan only, missing {missing}. "
                    "Add files that exist on disk, then /mix send again.")
        try:
            receipt = deliver(target, rendered, sender=sender,
                              as_voice_note=as_voice)
        except ValueError as exc:
            return f"couldn't send: {exc}"
        kind = "voice note" if receipt.kind == "voice_note" else "audio"
        return (f"sent as {kind} to {target} — {current.title} "
                f"({rendered.total_duration_s:.0f}s)")

    if verb == "share":
        return share_descriptor(current)

    return f"unknown /mix action '{verb}'\n\n" + mix_menu()
