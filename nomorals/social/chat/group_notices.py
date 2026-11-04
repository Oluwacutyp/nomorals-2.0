"""The notice layer (DESIGN §31) — one composer for every
automated group action.

A notice is the public record of an action Devon took on her
own: who it hit (mention + the numeric id, from the member
resolver's output), the trigger in plain words, the action with
its ladder state, the expiry when the action is time-bound —
and an Undo affordance that reverses exactly that action.  The
surface records the notice's STATE (`GroupSurface.record_notice`);
this module renders that state: the posted text and the buttons
are authored from the same record, so the button under a notice
can never disagree with the words above it.

Platform honesty, per the gateway contract: only the Telegram
family renders inline buttons, so on every other platform the
notice carries the equivalent text command instead (the one id
a human must quote back to act — the standing exception to the
no-internal-ids rule).  Where the platform supports editing, an
undo EDITS the notice in place; elsewhere the undo posts its
outcome as a short follow-up.
"""
from __future__ import annotations

from typing import Any

from .expression import button_label

#: Platforms whose adapters render inline keyboards (gateway
#: contract in social/chat/base.py).  Everything else gets the
#: text-command form of every affordance.
BUTTON_PLATFORMS = ("telegram-bot", "telegram")


def platform_has_buttons(platform: str) -> bool:
    return str(platform or "").lower() in BUTTON_PLATFORMS


def undo_command(notice_id: str) -> str:
    return f"undonotice {notice_id}"


def notice_keyboard(record: dict[str, Any],
                    settings: dict[str, Any] | None = None) -> list | None:
    """The buttons under a notice, from the record itself: an
    undo button while the action stands; nothing once undone
    (the edited text carries the outcome instead). The label
    comes from the one button vocabulary (§21) — the record's
    own presentation settings may override it."""
    if not record or record.get("undone"):
        return None
    if (record.get("extra") or {}).get("reversible") is False:
        return None
    return [[(button_label("undo", "Undo", settings=settings),
              undo_command(str(record.get("id"))))]]


def notice_text(record: dict[str, Any], platform: str) -> str:
    """The text a notice posts with, on ``platform``: the
    composed record text, plus the text form of the undo
    affordance where buttons don't render."""
    text = str(record.get("text") or "")
    if record.get("undone"):
        return text
    if (record.get("extra") or {}).get("reversible") is False:
        return text
    if not platform_has_buttons(platform):
        return (f"{text}\n↩️ undo: /{undo_command(record['id'])}"
                if text else
                f"↩️ undo: /{undo_command(record['id'])}")
    return text


def undone_text(record: dict[str, Any], outcome: str) -> str:
    """The notice after an undo: the original words stand (they
    were true when posted), with the reversal appended — an edit
    in place, never a stacked second record."""
    base = str(record.get("text") or "")
    line = f"↩️ **Undone** — {outcome}"
    return f"{base}\n{line}" if base else line


def drop_notice_text(mention: str, user_id: str, *,
                     total: int, per_claim: int,
                     treasury: bool) -> str:
    """The composed record line for a drop opening (the rich
    Card announcement stays the display; this is the notice
    record's text — what an edit-in-place would show)."""
    from .typography import sparks

    source = "the treasury" if treasury else mention
    return (f"🎁 {mention} [{user_id}] opened a drop: "
            f"**{sparks(total)}** from {source}, "
            f"{sparks(per_claim)} per claim.")
