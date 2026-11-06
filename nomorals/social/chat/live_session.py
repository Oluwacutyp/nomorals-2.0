"""Live sessions: ONE live message per interactive flow, edited in
place (presentation architecture, owner correction 2026-11-06).

The defect this organ kills: an interactive flow posted its opening
state as one message and every state change as a NEW message — the
riddle printed ``(/riddle answer reveals it)`` as raw text, and the
reveal arrived as a second post (``the answer was: …``). The owner's
law: interactive flows are ONE live message — actions are buttons on
it, and state changes EDIT that message in place.

The mechanism is declaration, not per-flow plumbing:

* A handler declares a render of the flow by returning
  ``actionable(text, buttons, session="<name>")`` (see
  :mod:`.tgbot_buttons`). A continuation render — the reveal, the
  next question — additionally passes ``session_edit=True``.
* The control-reply channel (``PartnerRuntime._send_control_reply``,
  the ONE funnel every command reply crosses) asks this module
  where the render belongs: :func:`edit_target` decides, the send
  records the live message in :class:`LiveSessionRegistry`, and a
  render that belongs in the live message goes out through
  :func:`edit_session_message` instead of a fresh send.

Edit targeting, in order:

1. The turn TAPPED the session's live message (the adapters stamp
   ``tapped_message_id`` into the synthetic callback message's
   meta; both WhatsApp bridges report the tapped message as the
   tap's ``reply_to``) → edit THAT message, wherever the tap
   landed, and it becomes the recorded live message.
2. The reply declares ``session_edit`` and a live message is
   recorded for the chat → edit the recorded message.
3. Otherwise → a fresh send; its message id becomes the record.

Surfaces degrade honestly, never fakely:

* A platform whose limits deny text editing
  (:mod:`.platform_limits` — SMS, local, Discord's adapter today)
  never edits: every render is a fresh send, exactly as before.
* An adapter edit verb that refuses (WhatsApp's ~15-minute window,
  a message the bridge no longer holds, Telegram's not-modified)
  falls back to a fresh send and the record moves to the new
  message. A failed edit is never reported as an edit.
* WhatsApp's edit verb is TEXT-ONLY (the protocol swaps no
  keyboard): the declared buttons of the edited render do not
  replace the original message's buttons there. Session keyboards
  are therefore authored state-invariant where the flow allows it
  (the riddle's Reveal/Another pair is correct in every state);
  Telegram's edit swaps the keyboard with the text.

This module never raises: a session failure degrades to the plain
send path, which is always available.
"""

from __future__ import annotations

import inspect
import logging
import threading
from typing import Any

_log = logging.getLogger(__name__)

__all__ = [
    "LiveSessionRegistry",
    "edit_target",
    "edit_session_message",
    "tapped_message_id",
]


class LiveSessionRegistry:
    """(chat key, session) → the platform message id of the one
    live message. In-memory by design: a restart forgets the map,
    the next render posts fresh and re-records — the flow degrades
    to one extra post, never to a broken state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: dict[tuple[str, str], str] = {}

    def record(self, chat_key: str, session: str, message_id: str) -> None:
        if not chat_key or not session or not message_id:
            return
        with self._lock:
            self._live[(str(chat_key), str(session))] = str(message_id)

    def lookup(self, chat_key: str, session: str) -> str:
        if not chat_key or not session:
            return ""
        with self._lock:
            return self._live.get((str(chat_key), str(session)), "")

    def forget(self, chat_key: str, session: str) -> None:
        with self._lock:
            self._live.pop((str(chat_key), str(session)), None)


def tapped_message_id(message: Any) -> str:
    """The platform id of the message this turn's button tap landed
    on, or "" when the turn was not a tap (or the surface cannot
    say). Telegram's two adapters stamp it into the synthetic
    callback message's meta; both WhatsApp bridges surface the
    tapped message as the tap message's ``reply_to`` (the button
    response's context stanza), which the caller compares against
    the recorded live message."""
    meta = getattr(message, "meta", None) or {}
    if isinstance(meta, dict):
        tapped = str(meta.get("tapped_message_id") or "")
        if tapped:
            return tapped
    return ""


def edit_target(registry: LiveSessionRegistry, message: Any,
                session: str, *, session_edit: bool) -> str:
    """The message id this session render should EDIT, or "" when
    it must go out as a fresh send. See the module docstring for
    the targeting order."""
    chat = getattr(message, "chat", None)
    chat_key = str(getattr(chat, "key", "") or "")
    if not chat_key or not session:
        return ""
    recorded = registry.lookup(chat_key, session)
    tapped = tapped_message_id(message)
    if not tapped and recorded:
        # WhatsApp shape: the tap carries no meta stamp; the
        # bridges report the tapped message as reply_to instead.
        reply_to = str(getattr(message, "reply_to", "") or "")
        if reply_to and reply_to == recorded:
            tapped = reply_to
    if tapped:
        return tapped
    if session_edit and recorded:
        return recorded
    return ""


def _edit_verdict(result: Any) -> bool:
    """One success rule across the adapters' edit shapes: the Bot
    API adapter returns the API dict (None on ANY failure, its
    documented contract); the userbot and WhatsApp adapters return
    verdict dicts carrying ``ok``."""
    if result is None:
        return False
    if isinstance(result, dict) and "ok" in result:
        return bool(result.get("ok"))
    return True


def edit_session_message(adapter: Any, chat: Any, message_id: str,
                         text: str, *, buttons: Any = None,
                         parse_mode: str = "") -> bool:
    """Edit one live session message through whichever edit verb
    the surface's adapter exposes. Returns True only when the
    edit landed. Never raises.

    The verb shapes differ per surface and each is called in its
    own grammar: Telegram's adapters take the declared buttons
    (their edits swap the keyboard with the text); WhatsApp's
    bridge edit is text-only, so no buttons are passed (the
    original message keeps the keyboard it was sent with); the
    Bot API adapter additionally takes the send's parse mode so
    the edited render formats exactly like the send it replaces.
    """
    verb = getattr(adapter, "edit_message", None)
    if not callable(verb) or not message_id:
        return False
    try:
        params = inspect.signature(verb).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic verb
        params = {}
    kwargs: dict[str, Any] = {}
    if "buttons" in params and buttons is not None:
        kwargs["buttons"] = buttons
    if "parse_mode" in params and parse_mode:
        kwargs["parse_mode"] = parse_mode
    try:
        result = verb(chat, str(message_id), text, **kwargs)
    except TypeError:
        # A verb whose real shape differs from its signature's
        # promise: one honest retry in the minimal grammar.
        try:
            result = verb(chat, str(message_id), text)
        except Exception as exc:  # noqa: BLE001 - degrade to send
            _log.debug("live_session: edit retry failed: %s", exc)
            return False
    except Exception as exc:  # noqa: BLE001 - degrade to send
        _log.debug("live_session: edit failed: %s", exc)
        return False
    return _edit_verdict(result)
