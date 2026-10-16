"""WhatsApp group digests.

A group the owner is in can produce hundreds of messages a day. The
digest answers "what needs me?" without the owner reading the scroll:

* **mentioned Nx** — messages that tag the owner (``@name`` / name
  mention), counted per group.
* **decisions needing you** — messages that ask a question or call for a
  decision (``?``, "should we", "decide", "vote", "confirm", "approve",
  "need you", "your call") and are not yet resolved.
* **resolved threads** — questions that got an answer in-thread
  ("resolved", "done", "sorted", "agreed", "✅" on a decision message).

Per-group policy (stored per group id, default ``only-when-mentioned``):

* ``always`` — digest every group, even with no mentions.
* ``only-when-mentioned`` — digest only when the owner was mentioned
  or a decision needs them.
* ``never`` — no digest for this group, ever.

Pure functions: messages are plain dicts
(``{"sender", "text", "ts", "reply_to"}``), so this is fully testable
without a WhatsApp session. The adapter supplies the messages; this
module decides what the owner sees.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "GroupPolicy",
    "GROUP_POLICIES",
    "DEFAULT_GROUP_POLICY",
    "classify_message",
    "build_digest",
    "should_digest",
]

#: Per-group digest policies.
GROUP_POLICIES: tuple[str, ...] = ("always", "only-when-mentioned", "never")

#: Default when a group has no explicit policy.
DEFAULT_GROUP_POLICY = "only-when-mentioned"

#: Alias type for readability.
GroupPolicy = str

_DECISION_RES = (
    re.compile(r"\?+\s*$"),
    re.compile(r"\b(should we|shall we|decide|decision|vote|confirm|approve|need you|your call|what do you think|any objection)\b", re.I),
)
_RESOLVED_RES = (
    re.compile(r"\b(resolved|sorted|settled|agreed|done|closed|fixed|confirmed)\b", re.I),
    re.compile(r"✅|✔️|👍"),
)
_MENTION_RES = (
    re.compile(r"@(\w+)"),
)


def classify_message(text: str, *, me: str = "") -> dict[str, bool]:
    """Flags for one message: mentioned / decision / resolved.

    ``me`` is the owner's name/username (``@`` optional). A message is
    "mentioned" when it tags ``@me`` or contains the owner's name as a
    word. "decision" is a question-or-call-for-decision that is not
    already marked resolved.
    """
    t = text or ""
    mentioned = False
    if me:
        name = me.strip().lstrip("@").lower()
        if name:
            if f"@{name}" in t.lower():
                mentioned = True
            elif re.search(rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])", t.lower()):
                mentioned = True
    decision = any(rx.search(t) for rx in _DECISION_RES)
    resolved = any(rx.search(t) for rx in _RESOLVED_RES)
    if resolved:
        decision = False  # resolved threads stop needing the owner
    return {"mentioned": mentioned, "decision": decision, "resolved": resolved}


def should_digest(policy: str, *, mentions: int, decisions: int) -> bool:
    """Whether this group earns a digest under its policy."""
    p = (policy or DEFAULT_GROUP_POLICY).strip().lower()
    if p == "never":
        return False
    if p == "always":
        return True
    # only-when-mentioned (and any unknown value fails closed to it)
    return mentions > 0 or decisions > 0


def build_digest(group: str, messages: list[dict[str, Any]], *,
                 policy: str = DEFAULT_GROUP_POLICY,
                 me: str = "",
                 max_items: int = 10) -> dict[str, Any]:
    """Build one group's digest. Returns a report dict::

        {"group", "policy", "included", "mentions", "decisions",
         "resolved", "text"}

    ``text`` is "" when the group earns no digest under its policy —
    callers send nothing instead of an empty digest.
    """
    mentions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    resolved_count = 0
    for msg in messages or []:
        flags = classify_message(str(msg.get("text", "")), me=me)
        if flags["mentioned"]:
            mentions.append(msg)
        if flags["decision"]:
            decisions.append(msg)
        if flags["resolved"]:
            resolved_count += 1
    included = should_digest(policy, mentions=len(mentions), decisions=len(decisions))
    if not included:
        return {"group": group, "policy": policy, "included": False,
                "mentions": len(mentions), "decisions": len(decisions),
                "resolved": resolved_count, "text": ""}
    lines = [f"💬 *{group}* — digest"]
    if mentions:
        lines.append(f"  👤 mentioned ×{len(mentions)}:")
        for m in mentions[:max_items]:
            lines.append(f"    • {m.get('sender', '?')}: {(str(m.get('text', ''))[:140])}")
    if decisions:
        lines.append(f"  ❓ decisions needing you ×{len(decisions)}:")
        for m in decisions[:max_items]:
            lines.append(f"    • {m.get('sender', '?')}: {(str(m.get('text', ''))[:140])}")
    if resolved_count:
        lines.append(f"  ✅ resolved threads ×{resolved_count}")
    if not mentions and not decisions:
        lines.append(f"  ({len(messages or [])} messages, nothing needs you)")
    return {"group": group, "policy": policy, "included": True,
            "mentions": len(mentions), "decisions": len(decisions),
            "resolved": resolved_count, "text": "\n".join(lines)}
