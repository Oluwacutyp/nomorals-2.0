"""Burn-note tool: ``burn_note`` — seal and read one-time notes.

Devon seals a note under a fresh random key and hands the link
back; the key travels inside the link and never lives in her
store. The first read returns the text and deletes the row — a
second read finds nothing. One tool, action-dispatched, over
:mod:`nomorals.memory.ephemeral_notes`; the organ owns the seal,
the store, the burn log and the sweep; this module is only the
surface.

``action=seal`` seals text (``text=...``, ``expire=1h|1d|7d|30d|never``
— default 30 days — ``password=...`` adds an extra gate) and
returns the link. ``action=read`` reads once (``code=...``,
``password=...`` when one was set) and returns the text — or an
honest refusal when the note is gone, expired, or the code or
password is wrong. ``action=list`` lists live notes as metadata
only, never contents. ``action=purge`` deletes an unread note
early. ``action=status`` reports the live count, the burn log
length and the sweep job.

The tool never logs a note's text or a link's key — neither is
ever passed to a logger anywhere in this path.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from ..memory.ephemeral_notes import (
    BURN_NOTE_JOB,
    BURN_NOTE_SPEC,
    BurnNoteStore,
    list_notes,
    parse_expiry,
    purge_note,
    read_note,
    seal_note,
)

__all__ = ["register"]


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def register(registry: Any) -> None:
    """Attach the ``burn_note`` tool."""
    context = registry.context

    @registry.register(
        "burn_note",
        description=(
            "Seal and read burn-after-reading notes: write a note, "
            "get a link, the first read shows it and destroys it — "
            "a second read finds nothing. action=seal seals text "
            "(text=<the note>, expire=<1h|1d|7d|30d|never, default "
            "30d>, password=<optional extra gate>) and returns the "
            "link — the key travels inside the link and never lives "
            "in storage. action=read reads once (code=<the link>, "
            "password=<when one was set>) and returns the text, or "
            "an honest refusal when the note is gone, expired, or "
            "the code or password is wrong. action=list lists live "
            "notes as metadata only, never contents. action=purge "
            "deletes an unread note early (note_id=<id>). "
            "action=status reports the live count, the burn log and "
            "the sweep job. Anyone holding the link can read it "
            "once — that is the design, not a bug."
        ),
        capability=Capability.MEM_WRITE,
    )
    def burn_note_tool(action: str = "seal", text: str = "",
                       code: str = "", expire: str = "",
                       password: str = "",
                       note_id: str = "") -> dict[str, Any]:
        verb = (action or "seal").strip().lower()
        home = _home_of(context)
        if home is None:
            raise ValueError("burn_note needs a home (settings.home)")
        store = BurnNoteStore(home)
        if verb == "seal":
            if not (text or "").strip():
                raise ValueError("burn_note seal needs text")
            try:
                seconds = parse_expiry(expire if expire else None)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            return seal_note(store, text, expires_in_s=seconds,
                             password=password or None)
        if verb == "read":
            if not (code or "").strip():
                raise ValueError("burn_note read needs a code")
            return read_note(store, code.strip(),
                             password=password or None)
        if verb == "list":
            return {"ok": True, "action": "list",
                    "notes": list_notes(store)}
        if verb == "purge":
            if not (note_id or "").strip():
                raise ValueError("burn_note purge needs a note_id")
            purged = purge_note(store, note_id.strip())
            return {"ok": purged, "action": "purge",
                    "note_id": note_id.strip(),
                    "error": "" if purged else "no such live note"}
        if verb == "status":
            return {"ok": True, "action": "status",
                    "live": len(list_notes(store)),
                    "burned": len(store.burned_log()),
                    "job": BURN_NOTE_JOB, "spec": BURN_NOTE_SPEC}
        raise ValueError(f"unknown burn_note action {verb!r} "
                         "(seal | read | list | purge | status)")
