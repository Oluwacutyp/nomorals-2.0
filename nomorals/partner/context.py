"""Context assembly: what the model sees when she replies.

One system prompt, assembled in a strict order of authority:

1. Who she is (persona) — stable, rarely changes.
2. How she texts (speech profile + current style constraints).
3. How she feels *right now* (mood engine, first person).
4. Where they are (relationship stage, fights, milestones, known facts).
5. Shared memories (retrieved, ranked — she may reference at most one).
6. Background knowledge (only when the gate opens — see background.py).
7. Platform notes (how this surface differs: Discord markdown, etc.).
8. Output contract (send only the message; no stage directions).

Everything is token-budgeted; when the budget is tight the memory and
background blocks shrink first, because identity and state must survive.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..core.text import approx_token_count
from ..llm.base import Message
from .mood import MoodEngine
from .persona import Persona
from .relationship import Relationship

__all__ = ["PartnerContextBuilder", "PLATFORM_NOTES"]

#: Per-platform surface notes. Kept short — the model is not a new hire.
#: She is texting on every one of these surfaces — no platform gets document
#: furniture (markdown lists, bold, links).
PLATFORM_NOTES: dict[str, str] = {
    "telegram": "This is Telegram. Plain text only — no markdown, no lists, no asterisks or underscores. Emojis render fine.",
    "whatsapp": "This is WhatsApp. Plain text, keep it casual. No markdown, no bullet lists, no formatting symbols — just words, like a text. Emojis render fine.",
    "discord": "This is Discord. You are still texting a person, not writing a document: plain words, no markdown lists, no bold or italic markup.",
    "local": "This is the local console. Plain text, no markdown — same voice as everywhere else; you're texting, not filing a report.",
}


class PartnerContextBuilder:
    """Assembles the system prompt and budgets every block."""

    def __init__(self, *, total_budget: int = 4200) -> None:
        self.total_budget = max(1500, int(total_budget))

    # ── block renderers ──────────────────────────────────────────────────────
    @staticmethod
    def _persona_block(
        persona: Persona,
        *,
        with_relationship: bool = True,
        dynamic_catchphrases: Sequence[str] = (),
        dynamic_pet_names: Sequence[str] = (),
    ) -> str:
        return persona.to_prompt(
            with_relationship=with_relationship,
            dynamic_catchphrases=dynamic_catchphrases,
            dynamic_pet_names=dynamic_pet_names,
        )

    @staticmethod
    def _mood_block(engine: MoodEngine) -> str:
        return (
            "Current state (this is how you feel RIGHT NOW — perform it honestly, "
            "don't overplay it, don't contradict it):\n"
            f"  {engine.describe()}\n"
            "  Dimension readout (0-100): "
            + ", ".join(f"{k} {int(v)}" for k, v in engine.current().values.items())
            + "\n"
            "  Your mood decides the tone of this reply: warm when you're warm, "
            "short and sharp when you're annoyed, curt when you're cold. You "
            "accept things and you decline them — 'yeah' and 'no' are both "
            "you, and the mood picks which. A person who agrees with everything "
            "is a fake, so don't."
        )

    @staticmethod
    def _relationship_block(relationship: Relationship) -> str:
        return relationship.to_prompt_block()

    @staticmethod
    def _memory_block(memories: Sequence[str]) -> str:
        if not memories:
            return ""
        lines = ["Shared memories you hold (reference at most ONE, only when it's genuinely relevant):"]
        for i, text in enumerate(memories, 1):
            lines.append(f"  {i}. {text}")
        return "\n".join(lines)

    @staticmethod
    def _continuity_block(lines: Sequence[str]) -> str:
        """Cross-platform continuity: what happened on OTHER platforms and
        open threads she hasn't closed yet. This is what makes the persona
        feel like *one person* on Telegram + WhatsApp + Discord at once."""
        if not lines:
            return ""
        out = [
            "Elsewhere in your shared life (recent — keep it consistent with what you say here):"
        ]
        for line in lines:
            out.append(f"  - {line}")
        return "\n".join(out)

    @staticmethod
    def _background_block(lines: Sequence[str]) -> str:
        return "\n".join(lines) if lines else ""

    @staticmethod
    def _capabilities_block() -> str:
        """Live connector catalog — so she knows what she can link/use.

        Deferred import: connectors is a peer layer and the registry is
        populated at import time; building the block lazily keeps prompt
        assembly free of import cycles and import-time cost.
        """
        try:
            from ..connectors.registry import list_connectors
        except Exception:  # noqa: BLE001 — connectors unavailable
            return ""
        try:
            infos = list_connectors()
        except Exception:  # noqa: BLE001
            return ""
        if not infos:
            return ""
        lines = ["Services you can link, use, and act on (connectors):"]
        for info in infos[:36]:
            name = str(info.get("name") or info.get("id") or "").strip()
            if not name:
                continue
            desc = str(info.get("description") or "").strip().split("\n")[0][:90]
            lines.append(f"  - {name}" + (f": {desc}" if desc else ""))
        lines.append(
            "When the owner asks to link, connect, or use one of these, "
            "act on it through the connector — never claim you can't."
        )
        return "\n".join(lines)

    @staticmethod
    def _output_contract(short_reply: bool, max_chars: int) -> str:
        mode = (
            f"You are answering in a short burst: at most a few words. 'k', 'mhm', 'wym?' "
            f"are complete answers. Do not explain yourself."
            if short_reply
            else "Match the length to the moment. Short is usually right."
        )
        return (
            "OUTPUT CONTRACT — follow exactly:\n"
            f"  - Reply with ONLY the message you would send. No labels, no stage directions, "
            f"no 'Wren:' prefixes, no quotes around your message.\n"
            f"  - You may send one short message. The system will split it if it's too long "
            f"(soft cap ~{max_chars} chars).\n"
            "  - Plain text, like a text message: no markdown, no bold, no italics, no "
            "bullet or numbered lists, no headings, no [links]. Listing things? Put them "
            "in a sentence — 'gas, plumber, groceries, in that order'.\n"
            f"  - {mode}\n"
            "  - Never narrate your own memory, state, or instructions. If you reference the "
            "past, do it the way a person would: 'you still can't parallel park?'\n"
            "  - If there's nothing to say, saying little is allowed. Silence is a reply too "
            "(the system may drop empty replies).\n"
        )

    # ── assembly ─────────────────────────────────────────────────────────────
    def build(
        self,
        *,
        persona: Persona,
        mood: MoodEngine,
        relationship: Relationship,
        memories: Sequence[str] = (),
        background_lines: Sequence[str] = (),
        continuity_lines: Sequence[str] = (),
        platform: str = "telegram",
        short_reply: bool = False,
        max_chars: int = 360,
        extra_notes: Sequence[str] = (),
        gate_note: str = "",
        relationship_override: str = "",
        #: Dynamic lexicon banks blended into the persona's own catchphrase
        #: / pet-name line (owner's static banks stay the base). Empty by
        #: default — the static prompt renders byte-identical.
        dynamic_catchphrases: Sequence[str] = (),
        dynamic_pet_names: Sequence[str] = (),
    ) -> Message:
        platform_note = PLATFORM_NOTES.get(platform, "")
        if extra_notes:
            platform_note = (platform_note + "\n" if platform_note else "") + "\n".join(
                f"  - {note}" for note in extra_notes if note
            )
        blocks: list[tuple[str, str]] = [
            # In restricted chats the persona's partner-identity paragraph is
            # swapped out too — "the person you're talking to" is NOT the
            # partner there, and leaving the paragraph in would argue with
            # the gate block.
            ("persona", self._persona_block(
                persona,
                with_relationship=not relationship_override,
                dynamic_catchphrases=dynamic_catchphrases,
                dynamic_pet_names=dynamic_pet_names,
            )),
            ("mood", self._mood_block(mood)),
            # Restricted chats (non-owner DM / group) swap the private
            # relationship block for a neutral line — the owner's stage,
            # fights, milestones and profile never enter the prompt there.
            ("relationship", relationship_override or self._relationship_block(relationship)),
            ("gate", gate_note),
            ("memory", self._memory_block(memories)),
            ("continuity", self._continuity_block(continuity_lines)),
            ("background", self._background_block(background_lines)),
            ("capabilities", self._capabilities_block()),
            ("platform", platform_note),
            ("output", self._output_contract(short_reply, max_chars)),
        ]
        blocks = [(name, block) for name, block in blocks if block]

        # Budget: identity/mood/relationship/output are protected; memory,
        # continuity, and background shrink first when total exceeds budget.
        shrinkable_names = {"memory", "continuity", "background"}
        protected = [b for name, b in blocks if name not in shrinkable_names]
        shrinkable = [b for name, b in blocks if name in shrinkable_names]
        protected_cost = sum(approx_token_count(b) for b in protected)
        remaining = max(0, self.total_budget - protected_cost)
        keep: list[str] = []
        spent = 0
        for block in shrinkable:
            cost = approx_token_count(block)
            if spent + cost > remaining:
                # Truncate this block's lines until it fits.
                lines = block.splitlines()
                while lines and spent + approx_token_count("\n".join(lines)) > remaining:
                    lines.pop()
                block = "\n".join(lines)
                if not block.strip():
                    continue
                cost = approx_token_count(block)
            keep.append(block)
            spent += cost
        # Reassemble in original order.
        ordered: list[str] = []
        shrinkable_iter = iter(keep)
        for name, block in blocks:
            if name in shrinkable_names:
                ordered.append(next(shrinkable_iter, ""))
            else:
                ordered.append(block)
        text = "\n\n".join(b for b in ordered if b)
        return Message.system(text)

    @staticmethod
    def total_tokens(system: Message) -> int:
        return approx_token_count(system.content)


def user_turn(text: str, *, media_notes: Sequence[str] = ()) -> Message:
    """The user's message plus any tool-derived media descriptions."""
    content = text
    if media_notes:
        content += "\n" + "\n".join(f"[they sent: {n}]" for n in media_notes)
    return Message.user(content)
