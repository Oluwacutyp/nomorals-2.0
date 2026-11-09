"""Channel-text exports: the /channeltexts builder half.

The owner's ask (2026-11-08): pull the texts of a specific
channel "since it's in the db", delivered as a FILE — the
in-chat reply carries only the coverage summary.  This module
is the pure half: parse the command tail, classify and window
the stored rows (the personal vault's channel kinds — see
PersonalVault.channel_texts), and render the export in txt /
md / csv.  The runtime handler owns resolution, the stores
and the send; everything here is deterministic over rows.

Honesty rules baked into the render: the header states the
store's actual coverage (first → last stored post, counts by
kind) next to the window actually exported, so a short file
because the archive is young never reads as a quiet channel;
media posts appear as entries (with their caption when one
was stored), missed posts appear as entries, and the owner's
own posts are marked as his — the file is the channel's whole
stored record for the window, never a silent subset.
"""

from __future__ import annotations

import csv
import io
import re
import time
from datetime import datetime
from typing import Any

#: The default export cap: a file, not a chat dump, but a
#: bound still — the summary names it when it bites.
DEFAULT_LIMIT = 1000

#: The telegram pull backfill depth when --pull carries no
#: number of its own.
DEFAULT_PULL = 100
MAX_PULL = 500

FORMATS = ("txt", "md", "csv")


def stamp(ts: float) -> str:
    """One stored timestamp, the house export format."""
    if not ts:
        return "unknown time"
    return time.strftime("%d %b %Y %H:%M", time.localtime(ts))


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(name or "")).strip("-")
    return slug.lower() or "channel"


def row_class(row: dict[str, Any]) -> str:
    """text | media | missed | own — the export's per-row kind."""
    kind = str(row.get("kind") or "")
    if kind == "channel_post_missed":
        return "missed"
    if kind == "channel_posted":
        return "own"
    if str(row.get("path") or "") \
            or str(row.get("media_kind") or "text") not in ("", "text"):
        return "media"
    return "text"


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts + span over a row set (stored or exported)."""
    out: dict[str, Any] = {"total": len(rows), "text": 0,
                           "media": 0, "missed": 0, "own": 0,
                           "first_ts": 0.0, "last_ts": 0.0}
    for row in rows:
        out[row_class(row)] += 1
        ts = float(row.get("ts") or 0)
        if ts:
            if not out["first_ts"] or ts < out["first_ts"]:
                out["first_ts"] = ts
            if ts > out["last_ts"]:
                out["last_ts"] = ts
    return out


def coverage_line(cov: dict[str, Any]) -> str:
    """'12 stored posts (9 text · 2 media · 1 missed · 0 yours),
    03 Nov 2026 09:14 → 08 Nov 2026 22:03' — the honest span."""
    if not cov["total"]:
        return "nothing stored"
    parts = [f"{cov['text']} text"]
    if cov["media"]:
        parts.append(f"{cov['media']} media")
    if cov["missed"]:
        parts.append(f"{cov['missed']} missed")
    if cov["own"]:
        parts.append(f"{cov['own']} yours")
    return (f"{cov['total']} stored post(s) ({' · '.join(parts)}), "
            f"{stamp(cov['first_ts'])} → {stamp(cov['last_ts'])}")


def _parse_date(value: str, *, end_of_day: bool) -> float:
    """YYYY-MM-DD (a bare date for --until means its end) or a
    fuller ISO stamp, local time.  Raises ValueError."""
    value = value.strip()
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        dt = datetime.strptime(value, "%d %b %Y")
    if end_of_day and len(value) <= 10:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.timestamp()


def parse_tail(tail: str) -> tuple[str, dict[str, Any], str]:
    """(channel query, options, error) for one /channeltexts
    tail.  Options ride as --flags anywhere in the tail; every
    other token is part of the (possibly multi-word) channel
    name.  Options: --days N · --since DATE · --until DATE ·
    --limit N · --format txt|md|csv · --pull [N] (Telegram:
    backfill the archive from the userbot's history first)."""
    opts: dict[str, Any] = {"days": 0, "since": 0.0, "until": 0.0,
                            "limit": DEFAULT_LIMIT, "format": "txt",
                            "pull": 0}
    words: list[str] = []
    tokens = str(tail or "").split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        low = tok.lower()
        if low in ("--days", "--since", "--until", "--limit",
                   "--format", "--pull"):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if low == "--pull" and (not nxt or nxt.startswith("--")):
                opts["pull"] = DEFAULT_PULL
                i += 1
                continue
            if not nxt:
                return "", opts, f"{low} needs a value."
            try:
                if low == "--days":
                    opts["days"] = max(0, int(nxt))
                elif low == "--limit":
                    opts["limit"] = max(1, int(nxt))
                elif low == "--pull":
                    opts["pull"] = max(1, min(MAX_PULL, int(nxt)))
                elif low == "--since":
                    opts["since"] = _parse_date(nxt, end_of_day=False)
                elif low == "--until":
                    opts["until"] = _parse_date(nxt, end_of_day=True)
                elif low == "--format":
                    fmt = nxt.lower().lstrip(".")
                    if fmt not in FORMATS:
                        return "", opts, (
                            f"format is txt, md or csv — "
                            f"“{nxt}” isn't one.")
                    opts["format"] = fmt
            except ValueError:
                return "", opts, f"couldn't read the value for {low}: “{nxt}”."
            i += 2
            continue
        if low.startswith("--"):
            return "", opts, f"unknown option “{tok}”."
        words.append(tok)
        i += 1
    query = " ".join(words).strip()
    if opts["days"]:
        since = time.time() - opts["days"] * 86400
        opts["since"] = max(opts["since"], since) if opts["since"] \
            else since
    return query, opts, ""


def window_desc(opts: dict[str, Any]) -> str:
    if opts.get("since") and opts.get("until"):
        return f"{stamp(opts['since'])} → {stamp(opts['until'])}"
    if opts.get("since"):
        return f"since {stamp(opts['since'])}"
    if opts.get("until"):
        return f"until {stamp(opts['until'])}"
    return "everything stored"


def filter_rows(rows: list[dict[str, Any]],
                opts: dict[str, Any]) -> list[dict[str, Any]]:
    """Window + cap over oldest→newest rows: the window keeps
    its span, the cap keeps the LATEST posts inside it."""
    out = rows
    if opts.get("since"):
        out = [r for r in out if float(r.get("ts") or 0) >= opts["since"]]
    if opts.get("until"):
        out = [r for r in out if float(r.get("ts") or 0) <= opts["until"]]
    limit = int(opts.get("limit") or 0)
    if limit and len(out) > limit:
        out = out[-limit:]
    return out


def _sender(row: dict[str, Any], channel_name: str) -> str:
    if row_class(row) == "own":
        return "You (posted via Devon)"
    return str(row.get("poster_name") or "") or channel_name


def _media_note(row: dict[str, Any]) -> str:
    cls = row_class(row)
    kind = str(row.get("media_kind") or "media")
    if cls == "missed":
        return "[media — could not be downloaded when the post was seen]"
    path = str(row.get("path") or "")
    if path:
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        return f"[{kind} — saved file: {name}]"
    return f"[{kind} — the post's media itself is not in the archive]"


def _entry_text(row: dict[str, Any]) -> str:
    """The post's printable body: its text (a media post's
    caption included) and/or its media note."""
    text = str(row.get("text") or "")
    if row_class(row) in ("media", "missed"):
        note = _media_note(row)
        return f"{note}\n{text}" if text else note
    return text


def render_export(rows: list[dict[str, Any]], *,
                  channel_name: str, platform: str,
                  channel_id: str, fmt: str,
                  stored: dict[str, Any], opts: dict[str, Any],
                  exported_ts: float) -> str:
    """The whole export file body in one format.  *rows* are
    the exported (windowed) rows; *stored* is the coverage of
    ALL stored rows for the channel — header honesty."""
    platform_name = "WhatsApp" if platform == "whatsapp" else "Telegram"
    cov = coverage(rows)
    header = [
        f"Channel texts — {channel_name}",
        f"Platform: {platform_name} · Channel: {channel_id}",
        f"Exported: {stamp(exported_ts)}",
        f"Stored coverage: {coverage_line(stored)}",
        f"This file: {cov['total']} post(s) · window: "
        f"{window_desc(opts)}",
    ]
    if platform == "telegram" and stored["total"]:
        header.append(
            "Note: this archive's first stored post is "
            f"{stamp(stored['first_ts'])} — anything the channel "
            "posted earlier was not stored (Telegram channel "
            "posts were noted and dropped before this archive "
            "existed).")
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["ts", "datetime", "sender", "kind",
                    "media_kind", "text", "source_id", "saved_file"])
        for row in rows:
            w.writerow([
                f"{float(row.get('ts') or 0):.0f}",
                stamp(float(row.get("ts") or 0)),
                _sender(row, channel_name), row_class(row),
                str(row.get("media_kind") or ""),
                str(row.get("text") or ""),
                str(row.get("source_id") or ""),
                str(row.get("path") or "")])
        # CSV stays pure CSV (one header row, one row per
        # post) — the coverage header lives in the txt/md
        # forms and in the in-chat summary for every form.
        return buf.getvalue()
    lines: list[str] = []
    if fmt == "md":
        lines.append(f"# {header[0]}")
        lines += [f"- {h}" for h in header[1:]]
        lines.append("")
        for row in rows:
            lines.append(f"## {stamp(float(row.get('ts') or 0))} — "
                         f"{_sender(row, channel_name)}")
            lines.append("")
            lines.append(_entry_text(row))
            lines.append("")
    else:
        lines += header
        lines.append("")
        lines.append("─" * 60)
        for row in rows:
            lines.append(f"[{stamp(float(row.get('ts') or 0))}] "
                         f"{_sender(row, channel_name)}")
            lines.append(_entry_text(row))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
