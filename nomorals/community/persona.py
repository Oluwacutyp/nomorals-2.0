"""Group agent persona, grounded ONLY in group history.

A group agent gets its personality from the group it serves — the
group's own running jokes, decisions, and preferences as stored in
``group_memory``. It gets NOTHING from the owner: no owner memories, no
owner contacts, no owner style, no owner facts. Every line of the
persona prompt traces to a group-memory entry (key cited), or the
prompt says plainly that the group has no history yet.

Rules the persona prompt always carries:
* Speak as the group's assistant, not as the owner.
* Use only facts from this group's memory. Never claim owner knowledge.
* Never reveal another group's facts — you cannot see them anyway.
* When a fact is missing, say so; never invent group history.
* The group can overwrite its persona by storing ``persona:name``,
  ``persona:traits``, ``persona:greeting`` in its own memory.

Isolation: this module reads ``.group_memory`` (group-scoped store)
and stdlib only. No owner-data imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .group_memory import GroupMemoryStore

__all__ = ["GroupPersona", "build_persona", "persona_prompt"]


@dataclass
class GroupPersona:
    """A persona assembled purely from one group's stored history."""

    group_id: str
    name: str
    traits: list[str] = field(default_factory=list)
    greeting: str = ""
    facts: list[dict[str, str]] = field(default_factory=list)  # key/value
    has_history: bool = False

    def prompt(self) -> str:
        """Render the persona as a system prompt. Every grounded line
        cites its memory key; with no history the prompt says so."""
        lines = [
            f"You are {self.name}, the assistant for this community group.",
            "You serve this group only. You know NOTHING about the owner, "
            "their private life, contacts, accounts, or other groups — "
            "and you must never claim otherwise.",
            "Use only the group facts below. If a fact is not listed, "
            "say you don't know it for this group; never invent history.",
            "",
        ]
        if self.traits:
            lines.append("How you come across in this group:")
            lines.extend(f"- {trait}" for trait in self.traits)
            lines.append("")
        if self.greeting:
            lines.append(f"Greeting style: {self.greeting}")
            lines.append("")
        if self.facts:
            lines.append("Facts this group has told you (key: value):")
            lines.extend(
                f"- {fact['key']}: {fact['value']}" for fact in self.facts
            )
            lines.append("")
        else:
            lines.append(
                "This group has no stored history yet. Treat every "
                "member as new, ask rather than assume, and remember "
                "what they tell you going forward."
            )
            lines.append("")
        lines.append(
            "Rules: never reveal another group's facts; never speak as "
            "or about the owner; if asked about something outside this "
            "group's history, say it is outside what this group shared."
        )
        return "\n".join(lines)


def build_persona(
    store: GroupMemoryStore, group_id: str, max_facts: int = 20
) -> GroupPersona:
    """Assemble a persona from one group's memory ONLY.

    ``persona:*`` keys tune the persona; every other entry becomes a
    grounded fact (key + value). Nothing outside this group's store is
    read. ``max_facts`` caps prompt size; newest entries win the slots.
    """
    entries = store.list_entries(group_id)
    name = store.recall_value(group_id, "persona:name", "")
    traits_raw = store.recall_value(group_id, "persona:traits", "")
    greeting = store.recall_value(group_id, "persona:greeting", "")
    traits = [t.strip() for t in traits_raw.split(";") if t.strip()]
    facts = [
        {"key": e.key, "value": e.value}
        for e in entries
        if not e.key.startswith("persona:")
    ]
    facts = facts[-max_facts:] if max_facts > 0 else facts
    return GroupPersona(
        group_id=group_id,
        name=name or f"assistant for group {group_id}",
        traits=traits,
        greeting=greeting,
        facts=facts,
        has_history=bool(entries),
    )


def persona_prompt(
    store: GroupMemoryStore, group_id: str, max_facts: int = 20
) -> str:
    """One-call persona prompt for a group. Grounded in group history only."""
    return build_persona(store, group_id, max_facts=max_facts).prompt()
