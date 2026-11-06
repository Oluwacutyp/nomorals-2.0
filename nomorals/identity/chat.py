"""Chat control for /iddocs — one front door over nomorals.identity.

Never raises. Owner-only (the runtime gates the command; the tracker
itself also refuses non-owner calls).

Usage:
    /iddocs list
    /iddocs add <type> <expiry YYYY-MM-DD> <vault:ref> [country]
    /iddocs expiring [days]
    /iddocs overdue
    /iddocs renew <id>

Only vault *references* are stored (``vault:...``) — never a raw
document number; the tracker rejects anything else. Document ids are
printed because the owner must quote one back to act on it (renew) —
the designed quote-back exception to the no-internal-ids rule.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.tool_render import describe_exc

_log = get_logger(__name__)

__all__ = ["control_iddocs", "IDDOCS_USAGE"]


def IDDOCS_USAGE() -> str:
    return (
        "🪪 /iddocs — identity document expiry tracker (owner-only)\n"
        "  /iddocs list — every tracked document, soonest expiry first\n"
        "  /iddocs add <passport|visa|licence|id_card|permit|other> "
        "<expiry YYYY-MM-DD> <vault:reference> [country]\n"
        "  /iddocs expiring [days] — what needs action soon (default 90)\n"
        "  /iddocs overdue — already expired\n"
        "  /iddocs renew <id> — the renewal checklist for a document\n"
        "Only the vault reference is stored — never the document number."
    )


def _tracker(context: Any) -> Any:
    """IdentityTracker on the durable home path (SQLite)."""
    from .documents import IdentityTracker

    settings = getattr(context, "settings", None)
    home = (getattr(settings, "home_path", None)
            or getattr(settings, "workspace_dir", None))
    base = Path(home) if home else Path.cwd()
    path = base / "data" / "identity_documents.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return IdentityTracker(str(path))


def _fmt_doc(doc: Any, now: float) -> str:
    expiry = datetime.fromtimestamp(doc.expiry_ts).date().isoformat()
    days = int((doc.expiry_ts - now) // 86400)
    if days < 0:
        when = f"EXPIRED {-days}d ago"
    elif days == 0:
        when = "expires today"
    else:
        when = f"{days}d left"
    country = f" · {doc.issuing_country}" if doc.issuing_country else ""
    return (f"  {doc.id}: {doc.doc_type}{country} — expires {expiry} "
            f"({when}) · {doc.number_ref}")


def control_iddocs(tail: str, context: Any = None, chat: Any = None,
                   sender_id: str = "", sender: str = "",
                   chat_key: str = "") -> str:
    """``/iddocs`` — identity document tracker surface. Never raises."""
    try:
        return _control_iddocs(tail or "", context)
    except Exception as exc:  # noqa: BLE001 — never raise from chat
        _log.warning("iddocs failed", exc_info=True)
        return f"Identity documents hit an error ({describe_exc(exc)})."


def _control_iddocs(tail: str, context: Any) -> str:
    raw = tail.strip()
    if not raw or raw.lower() == "help":
        return IDDOCS_USAGE()
    verb, _, rest = raw.partition(" ")
    verb = verb.lower()
    rest = rest.strip()
    tracker = _tracker(context)
    try:
        now = time.time()
        if verb in ("list", ""):
            docs = tracker.list(owner=True)
            if not docs:
                return ("no identity documents tracked yet.\n"
                        + IDDOCS_USAGE())
            lines = [f"🪪 identity documents ({len(docs)}):"]
            lines.extend(_fmt_doc(d, now) for d in docs)
            return "\n".join(lines)
        if verb == "add":
            return _add(tracker, rest)
        if verb == "expiring":
            days = 90.0
            if rest:
                try:
                    days = float(rest.split()[0])
                except ValueError:
                    return "usage: /iddocs expiring [days]"
            docs = tracker.expiring_within(days, owner=True)
            if not docs:
                return f"nothing expires within {days:.0f} days. ✅"
            lines = [f"⏰ expiring within {days:.0f} days ({len(docs)}):"]
            lines.extend(_fmt_doc(d, now) for d in docs)
            lines.append("/iddocs renew <id> for a renewal checklist.")
            return "\n".join(lines)
        if verb == "overdue":
            docs = tracker.overdue(owner=True)
            if not docs:
                return "nothing overdue. ✅"
            lines = [f"🔴 overdue documents ({len(docs)}):"]
            lines.extend(_fmt_doc(d, now) for d in docs)
            return "\n".join(lines)
        if verb == "renew":
            return _renew(tracker, rest)
        return IDDOCS_USAGE()
    finally:
        tracker.close()


def _add(tracker: Any, rest: str) -> str:
    from .documents import IdentityDocument

    parts = rest.split(None, 3)
    if len(parts) < 3:
        return ("usage: /iddocs add <type> <expiry YYYY-MM-DD> "
                "<vault:reference> [country]")
    doc_type, date_text, number_ref = parts[0], parts[1], parts[2]
    country = parts[3] if len(parts) > 3 else ""
    try:
        expiry = datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError:
        return f"expiry must be YYYY-MM-DD — got {date_text!r}."
    try:
        doc = IdentityDocument(
            owner_id="owner", doc_type=doc_type,
            issuing_country=country, number_ref=number_ref,
            expiry_ts=expiry.timestamp())
        tracker.add(doc, owner=True)
    except (ValueError, TypeError) as exc:
        return f"couldn't track that document: {exc}"
    return (f"tracked 🪪 {doc.doc_type} (id {doc.id}) — expires "
            f"{date_text}. /iddocs expiring watches it from here.")


def _renew(tracker: Any, rest: str) -> str:
    ref = rest.strip()
    if not ref:
        return "usage: /iddocs renew <id> (ids are in /iddocs list)"
    doc = tracker.get(ref, owner=True)
    if doc is None:
        matches = [d for d in tracker.list(owner=True)
                   if d.id.startswith(ref)]
        if len(matches) == 1:
            doc = matches[0]
    if doc is None:
        return f"no identity document matches {ref!r} — /iddocs list."
    steps = tracker.renewal_checklist(doc, owner=True)
    expiry = datetime.fromtimestamp(doc.expiry_ts).date().isoformat()
    lines = [f"🔁 renewal — {doc.doc_type} (expires {expiry}):"]
    for step in steps:
        lines.append(f"  {step['order']}. {step['title']} — {step['detail']}")
    return "\n".join(lines)
