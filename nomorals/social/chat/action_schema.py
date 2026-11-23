"""The rich action schema (Living Chat P1, 2026-11-08).

The keyboard path historically carried ``(label, data)`` pairs —
too small for an expressive interaction system. An :class:`Action`
carries the full description:

* **stable identity** — explicit ``id`` or derived deterministically
  from intent + payload, so the same action is the same action on
  every render and every surface;
* **intent** — callback / url / copy / choose / navigate / upload /
  refresh, declared rather than inferred from the payload's shape;
* **style** and **icon_custom_emoji_id** — Telegram's Bot API 9.4
  (2026-02-09) button fields, carried in the schema and HONORED
  on the Bot API surfaces: ``style`` colors the button
  (``primary`` blue / ``success`` green / ``danger`` red) and
  ``icon_custom_emoji_id`` shows a custom emoji before the
  button text. Both are gated the same way custom emoji in
  message text are — the bot owner has Telegram Premium, or the
  bot bought Fragment usernames — and the adapter retries a
  refused send with plain buttons instead of eating the message;
* **fallback_label** — the accessibility / fallback text used when
  the action degrades to a numbered list or a text line;
* **target / operation_id / revision** — the object and operation
  the action belongs to, for stateful cards.

Platform support is EVIDENCE, not aspiration. The Bot API
adapter (``TelegramBotAdapter._button_json``) serializes
``style`` + ``icon_custom_emoji_id`` onto callback AND url
buttons, so both are recorded **native** on ``telegram-bot`` /
``telegram-personal``; the in-text weaver carries registered
glyphs as ``custom_emoji`` entities through the same send path.
The Telethon adapter (MTProto ``Button.inline`` / ``Button.url``)
has no style or custom-emoji icon field, so both stay
**unsupported** there — resolution names them as dropped. The
WhatsApp bridges (Go ``messages.go`` cmdSend, Node
``whatsapp-bridge.mjs`` cmdSend) render callback actions as
quick-reply / list interactive messages and re-enter taps as
commands; they have no URL-button or per-button style/icon form,
so a URL action resolves to a text line there. Platforms with no
button concept resolve callback actions to the numbered fallback
— a peer path with real handlers, not a degraded afterthought.

An ``icon_custom_emoji_id`` is only ever sent when it names a
custom emoji Devon may actually use: ``resolve_action`` takes
the ids her registry holds (``known_custom_emoji_ids``) and
drops an id that is not among them by name — a bad id fails the
whole ``sendMessage``, so it is never sent blind. The sets
themselves attach through the existing expression path:
``/customemoji import <pack name>`` reads a whole readable pack
(any bot may READ a sticker set, even though SENDING its
custom emoji is eligibility-gated) and ``/customemoji add`` /
``ref`` curate entries by hand; created art registers in her
own packs through ``create_or_extend_custom_emoji_pack``.

Legacy tuples keep working everywhere: :func:`Action.to_legacy`
lowers an action to the ``(label, data)`` pair the existing
keyboards carry, and ``tgbot_buttons.actionable`` accepts Action
objects mixed with tuples.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .tgbot_buttons import check_callback_data

__all__ = [
    "INTENTS",
    "CALLBACK_INTENTS",
    "BUTTON_STYLES",
    "FIELD_SUPPORT",
    "SUPPORT_NOTES",
    "Action",
    "ResolvedAction",
    "resolve_action",
    "actions_keyboard",
]

#: Every intent an action may declare.
INTENTS = ("callback", "url", "copy", "choose", "navigate",
           "upload", "refresh")

#: Intents whose payload is callback data routed to a handler.
CALLBACK_INTENTS = ("callback", "choose", "navigate", "upload",
                    "refresh")

#: The Bot API 9.4 button styles — the closed set. Anything else
#: is not a style Telegram knows, so the schema refuses it at
#: construction instead of letting it fail a live send.
BUTTON_STYLES = ("primary", "success", "danger")


def _is_custom_emoji_id(value: str) -> bool:
    """A plausible Bot API custom-emoji id: the decimal form of
    the int64 Telegram hands back from ``getStickerSet`` /
    ``getCustomEmojiStickers`` (``"5368324170671202286"``)."""
    value = str(value or "")
    return bool(value) and value.isdigit() and len(value) <= 19

#: Field support per platform: "native" (the field reaches the
#: user), "fallback" (the action still works through the fallback
#: form), "unsupported" (declared but NOT honored — resolution
#: reports it dropped, by name).
FIELD_SUPPORT: dict[str, dict[str, str]] = {
    "telegram-bot": {
        "callback": "native", "url": "native", "copy": "unsupported",
        "style": "native", "icon_custom_emoji_id": "native",
    },
    "telegram-personal": {
        "callback": "native", "url": "native", "copy": "unsupported",
        "style": "native", "icon_custom_emoji_id": "native",
    },
    "telegram": {
        "callback": "native", "url": "native", "copy": "unsupported",
        "style": "unsupported", "icon_custom_emoji_id": "unsupported",
    },
    "whatsapp": {
        "callback": "native", "url": "fallback", "copy": "fallback",
        "style": "unsupported", "icon_custom_emoji_id": "unsupported",
    },
    "discord": {
        "callback": "fallback", "url": "fallback", "copy": "fallback",
        "style": "unsupported", "icon_custom_emoji_id": "unsupported",
    },
    "sms": {
        "callback": "fallback", "url": "fallback", "copy": "fallback",
        "style": "unsupported", "icon_custom_emoji_id": "unsupported",
    },
    "web": {
        "callback": "fallback", "url": "fallback", "copy": "fallback",
        "style": "unsupported", "icon_custom_emoji_id": "unsupported",
    },
    "local": {
        "callback": "fallback", "url": "fallback", "copy": "fallback",
        "style": "unsupported", "icon_custom_emoji_id": "unsupported",
    },
    "webhook": {
        "callback": "fallback", "url": "fallback", "copy": "fallback",
        "style": "unsupported", "icon_custom_emoji_id": "unsupported",
    },
}

#: Why the support table says what it says — the adapter/bridge
#: evidence, so a future reader can re-verify instead of trusting.
SUPPORT_NOTES: dict[str, str] = {
    "telegram-bot": (
        "TelegramBotAdapter._button_json emits style and "
        "icon_custom_emoji_id (Bot API 9.4, 2026-02-09) on "
        "callback and url buttons; the in-text weaver carries "
        "registered glyphs as custom_emoji entities. Both are "
        "gated on the bot owner having Telegram Premium or the "
        "bot having bought Fragment usernames — a refusal "
        "retries the send plain, never eats it."),
    "telegram-personal": (
        "Same Bot API adapter as telegram-bot (second bot)."),
    "telegram": (
        "Telethon adapter builds Button.inline / Button.url only; "
        "MTProto buttons carry no style or custom-emoji icon "
        "field, so both stay unsupported here even though the "
        "Bot API surfaces honor them."),
    "whatsapp": (
        "Both bridges (Go messages.go cmdSend, Node "
        "whatsapp-bridge.mjs cmdSend) render the first buttons as "
        "a NativeFlow quick-reply set or a ListMessage and re-enter "
        "taps as commands; no URL-button or per-button style/icon "
        "form exists, and a refused interactive send degrades to "
        "text action lines the adapter journals."),
    "discord": (
        "PlatformLimits: no button support in this adapter; "
        "actions ride the numbered fallback (button-fallback "
        "surface), which executes the same commands."),
    "sms": "No buttons; numbered/text fallback only.",
    "web": "No native keyboard in this adapter; numbered/text fallback.",
    "local": "No native keyboard; numbered/text fallback.",
    "webhook": "No native keyboard; numbered/text fallback.",
}

_TELEGRAM_PLATFORMS = ("telegram-bot", "telegram-personal", "telegram")


def _derive_id(intent: str, payload: str) -> str:
    digest = hashlib.sha256(
        f"{intent}\x1f{payload}".encode()).hexdigest()[:10]
    return f"{intent}:{digest}"


@dataclass(frozen=True)
class Action:
    """One interactive action, fully described."""

    label: str
    intent: str = "callback"
    data: str = ""                #: callback payload (the command)
    url: str = ""                 #: for intent="url"
    emoji: str = ""               #: ordinary emoji cue in the label
    style: str = ""               #: declared button style (see support)
    icon_custom_emoji_id: str = ""  #: native custom-emoji icon id
    fallback_label: str = ""      #: accessibility / fallback text
    target: str = ""              #: target object identity
    operation_id: str = ""        #: owning operation, when stateful
    revision: int = 0             #: state revision the action was cut at
    id: str = field(default="")

    def __post_init__(self) -> None:
        if self.intent not in INTENTS:
            raise ValueError(f"unknown action intent {self.intent!r}")
        if self.intent in CALLBACK_INTENTS:
            if not self.data:
                raise ValueError(
                    f"{self.intent} action {self.label!r} needs data")
            check_callback_data(self.data)
        if self.intent == "url" \
                and not self.url.startswith(("http://", "https://")):
            raise ValueError(
                f"url action {self.label!r} needs an http(s) url")
        if self.style and self.style not in BUTTON_STYLES:
            raise ValueError(
                f"unknown button style {self.style!r} — the Bot API "
                f"knows {', '.join(BUTTON_STYLES)}")
        if self.icon_custom_emoji_id \
                and not _is_custom_emoji_id(self.icon_custom_emoji_id):
            raise ValueError(
                "icon_custom_emoji_id must be Telegram's decimal id "
                f"form, got {self.icon_custom_emoji_id!r}")
        if not self.id:
            payload = self.url if self.intent == "url" else self.data
            object.__setattr__(
                self, "id", _derive_id(self.intent, payload))

    @property
    def effective_fallback_label(self) -> str:
        return self.fallback_label or self.label

    def to_legacy(self) -> tuple[str, str]:
        """The (label, data) pair the existing keyboards carry."""
        if self.intent == "url":
            return (self.label, self.url)
        return (self.label, self.data)


@dataclass(frozen=True)
class ResolvedAction:
    """How one action lands on one platform.

    ``form``: inline (Telegram keyboard) / interactive (WhatsApp
    quick-reply or list) / numbered (the numbered fallback) /
    text-line (the action rides as a text line). ``honored`` names
    the schema fields that reach the user; ``dropped`` names the
    declared fields that do NOT — never silently.
    """

    form: str
    honored: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    note: str = ""


def resolve_action(action: Action, platform: str,
                   known_custom_emoji_ids: Any = None
                   ) -> ResolvedAction:
    """Resolve an action against a platform's real support.

    ``known_custom_emoji_ids`` is the set of custom-emoji ids
    Devon's registry actually holds (her packs + imports — see
    the module docstring). When it is given, an
    ``icon_custom_emoji_id`` outside it is dropped by name as
    unverified: a bad id fails the whole ``sendMessage``, so it
    is never sent blind. When it is not given, a declared icon
    is honored as declared and the adapter's send-time retry
    ladder is the guard.
    """
    plat = str(platform or "")
    support = FIELD_SUPPORT.get(plat) or {
        "callback": "fallback", "url": "fallback", "copy": "fallback",
        "style": "unsupported", "icon_custom_emoji_id": "unsupported",
    }
    note = SUPPORT_NOTES.get(plat, "unknown platform: fallback forms only")
    known = None
    if known_custom_emoji_ids is not None:
        known = {str(i) for i in known_custom_emoji_ids if str(i)}
    style_ok = bool(action.style) \
        and support["style"] != "unsupported"
    icon_ok = bool(action.icon_custom_emoji_id) \
        and support["icon_custom_emoji_id"] != "unsupported"
    if icon_ok and known is not None \
            and action.icon_custom_emoji_id not in known:
        icon_ok = False
        note = (f"{note} icon_custom_emoji_id "
                f"{action.icon_custom_emoji_id!r} is not one of the "
                "custom emoji ids Devon holds (unverified — never "
                "sent blind, a bad id fails the whole send).")
    dropped: list[str] = []
    if action.style and not style_ok:
        dropped.append("style")
    if action.icon_custom_emoji_id and not icon_ok:
        dropped.append("icon_custom_emoji_id")
    rich: tuple[str, ...] = (
        (("style",) if style_ok else ())
        + (("icon_custom_emoji_id",) if icon_ok else ()))

    if plat in _TELEGRAM_PLATFORMS:
        if action.intent == "url":
            return ResolvedAction("inline",
                                  ("label", "url", "intent") + rich,
                                  tuple(dropped), note)
        if action.intent == "copy":
            return ResolvedAction(
                "text-line", ("label",),
                tuple(dropped + ["copy"]),
                "no copy_text button is emitted by the adapter; "
                "the text itself carries the content")
        return ResolvedAction("inline",
                              ("label", "data", "intent") + rich,
                              tuple(dropped), note)
    if plat == "whatsapp":
        if action.intent in ("url", "copy"):
            return ResolvedAction(
                "text-line", ("label", "fallback_label"),
                tuple(dropped + [action.intent]),
                "the bridges have no URL/copy button form; the "
                "action rides as a text line")
        return ResolvedAction(
            "interactive", ("label", "data", "intent"), tuple(dropped),
            "quick-reply set (first 3) or ListMessage form; "
            "overflow rides as text action lines")
    # No native buttons: the numbered fallback is the peer path.
    if action.intent == "url":
        return ResolvedAction("text-line", ("label", "url"),
                              tuple(dropped), note)
    return ResolvedAction("numbered", ("label", "data", "intent"),
                          tuple(dropped), note)


def actions_keyboard(rows: list) -> list[list[tuple[str, str]]]:
    """Lower rows of Action | (label, data) to a legacy keyboard."""
    out: list[list[tuple[str, str]]] = []
    for row in rows or []:
        lowered = []
        for item in row or []:
            if isinstance(item, Action):
                lowered.append(item.to_legacy())
            else:
                label, data = item
                lowered.append((str(label), str(data)))
        if lowered:
            out.append(lowered)
    return out
