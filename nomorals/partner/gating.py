"""Ownership-aware chat gating: who gets the full version of her.

The companion runs one persona on every surface, but NOT every surface
gets the full version:

* **owner DM / console** — full persona, full context: relationship
  stage, shared memories, continuity, background, everything.
* **private chat (a DM with someone who isn't the owner)** — she is the
  same person, texting someone she's getting to know: friendly, short,
  guarded. The owner's relationship, private facts, and shared memories
  are NOT in the prompt at all, so nothing leaks by accident.
* **group / channel** — one voice in the room: short, in the moment, no
  private details, no assistant behavior toward the group.

Classification is pure and cheap; the prompt blocks this module produces
are written in the same voice as the rest of the context (second person,
plain words, no document furniture). The owner's explicit steering still
works anywhere: an owner message in a group is classified as ``owner``,
and control commands were already operator-gated at the runtime.
"""

from __future__ import annotations

from ..social.chat.base import ChatKind, is_owner_chat

__all__ = [
    "MODE_OWNER",
    "MODE_PRIVATE",
    "MODE_GROUP",
    "classify_chat",
    "is_owner_chat",  # re-exported from nomorals.social.chat.base (L4)
    "gate_decision",
    "is_restricted",
    "gate_block",
    "relationship_block_for",
]

MODE_OWNER = "owner"
MODE_PRIVATE = "private"  # DM with someone who is not the owner
MODE_GROUP = "group"      # groups and channels (non-owner voice)


def classify_chat(chat, *, is_owner: bool) -> str:
    """Which version of her this message gets.

    ``chat`` is anything with ``.kind`` (a ChatRef). Channels classify as
    group (the brain never generates for a channel anyway — it's
    read-only — but the mode must be restricted if it ever did).
    """
    if is_owner:
        return MODE_OWNER
    if chat.kind == ChatKind.DM:
        return MODE_PRIVATE
    return MODE_GROUP


def gate_decision(chat, *, is_owner: bool, restricted_enabled: bool = True) -> str:
    """One gating decision for every surface.

    Wraps :func:`classify_chat` with the owner's kill-switch
    (``settings.partner.gate_restricted_chats``): when restricted gating
    is disabled everything is treated as owner mode. Every reply path
    must go through this — never inline the kill-switch check — so DM /
    group / stranger handling stays uniform across platforms.
    """
    raw_mode = classify_chat(chat, is_owner=is_owner)
    return raw_mode if restricted_enabled else MODE_OWNER


def is_restricted(mode: str) -> bool:
    return mode != MODE_OWNER


def gate_block(mode: str, *, platform: str = "", first_contact: bool = False) -> str:
    """The mode instructions, in her voice. Empty for the owner.

    ``first_contact`` adds the stranger frame for private chats: the first
    message from someone unknown gets a natural introduction beat, not the
    getting-to-know frame (you can't be "getting to know" someone you just
    met — first you figure out who they are).
    """
    if mode == MODE_OWNER:
        return ""
    if mode == MODE_PRIVATE:
        base = (
            "Who you're talking to RIGHT NOW: a DM with someone who is not the "
            "person you're with. Be exactly yourself — same voice, same mood — "
            "but the way a real person texts someone they're getting to know: "
            "friendly, easy, a little guarded.\n"
            "  - Nothing private goes here: no details about your relationship, "
            "who you're seeing, your home, your schedule, your work, your money, "
            "your day. You simply don't say that to them, the way you wouldn't "
            "say it to an acquaintance in person.\n"
            "  - You are not their assistant. No tasks, no research, no "
            "explaining how you work, no help-center voice. If they ask you to "
            "do something for them, answer the way a busy person would — a "
            "short, honest 'lol i can't' or a deflection — and move on.\n"
            "  - Keep it light and short. If it starts to push toward your "
            "private life, deflect in one casual line and change the subject."
        )
        if first_contact:
            base += (
                "\n  - This is the FIRST message from this person — you don't "
                "know them yet. Be natural about that: a brief hello in your "
                "own voice, a light read of what they said, maybe one easy "
                "question back. Don't assume anything about them, don't "
                "overshare to fill the silence, and don't interrogate them."
            )
        return base
    # group (and any non-owner channel voice)
    return (
        "Who you're talking to RIGHT NOW: a group chat with other people in it. "
        "You are one voice in the room, not the center of it.\n"
        "  - Speak in short, casual lines — the way people actually talk in "
        "group chats. No paragraphs, no lists, no explanations, no sign-offs.\n"
        "  - Nothing private goes here: no relationship talk, no home, schedule, "
        "work, or personal details. That stays with the person you're with.\n"
        "  - You are not the group's assistant. If someone in the group asks "
        "you to do something, answer like a person in the chat would — not like "
        "support. A short, casual answer is the whole answer."
    )


def relationship_block_for(mode: str) -> str:
    """The relationship block, replaced in restricted chats.

    The real block contains the owner's relationship stage, fights,
    milestones and profile — private by definition. In a non-owner chat it
    is swapped for this neutral line so the model knows whose DM it is
    without holding the private facts.
    """
    if mode == MODE_OWNER:
        return ""
    return (
        "Where you are in THIS chat: this person is not the one you're with. "
        "You are just being yourself — a real, busy person with your own life "
        "and your own people. You don't talk about your relationship here at "
        "all, and you don't share anything that belongs to your private life."
    )
