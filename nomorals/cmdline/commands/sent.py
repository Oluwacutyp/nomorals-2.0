"""``nm sent`` — delivery receipts from the durable chat outbox (G4)."""

from __future__ import annotations

import argparse
import time
from typing import Any

from ..emit import _emit


def _cmd_sent(args: argparse.Namespace, context: Any) -> int:
    """``nm sent --to <chat> --since <t>`` — delivery proof for chat sends.

    Prints one line per outbox row: state, attempts, message_id, error.
    Receipts only exist for durable sends (gateway ``durable_outbox`` or
    per-send ``durable=True``); direct sends leave no row, and the
    command says so instead of printing a misleading empty table.
    """
    from ...social.chat.outbox import Outbox, parse_since

    to = (getattr(args, "to", "") or "").strip()
    try:
        since = parse_since(getattr(args, "since", "24h"))
    except ValueError as exc:
        _emit(args, {"error": str(exc)}, f"sent: {exc}")
        return 2
    try:
        limit = max(1, int(getattr(args, "limit", 50) or 50))
    except (TypeError, ValueError):
        limit = 50
    outbox = Outbox(context.db)
    rows = outbox.receipts(chat_key=to, since=since, limit=limit)
    payload = {"receipts": rows, "count": len(rows), "to": to, "since": since}
    if not rows:
        _emit(args, payload,
              "no delivery receipts"
              + (f" for {to}" if to else "")
              + " — receipts only exist for durable sends "
                "(gateway durable_outbox, or durable=True)")
        return 0
    lines = []
    for row in rows:
        when = time.strftime("%Y-%m-%d %H:%M:%S",
                             time.localtime(row["created_at"]))
        line = (f"{when}  {row['state']:<16} attempts={row['attempts']}  "
                f"{row['chat_key']}  id={row['message_id'] or '-'}")
        if row["error"]:
            line += f"  error={row['error'][:160]}"
        lines.append(line)
    _emit(args, payload, "\n".join(lines))
    return 0
