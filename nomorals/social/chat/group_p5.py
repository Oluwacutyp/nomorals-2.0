"""The §16 P5 layer's pure pieces (DESIGN §16, census gap #9).

P1–P4 built the group core; §16's own completeness sweep named
a tail of platform capabilities that were never wired: member
tags, sender-chat bans, the join-request DM window + "queue"
verdict, reaction wipes, the forum-topic auto-rename duty,
ephemeral command answers, per-member command scopes, and
boost intel.  The adapters carry the platform verbs and the
runtime carries the commands; this module holds the pieces
that are pure — the settings shape, tag law, topic-name
derivation, and the boost-source vocabulary — so the surface
and the tests share one definition.

Honesty rules, matching the rest of the group build:

* A tag is presentation, not power: it never changes what a
  member may do, and it renders as literal text (a hostile tag
  is text, never markup).
* Boost intel records what Telegram's boost updates actually
  said.  A boost with no user in its source (an unclaimed
  giveaway code) is counted for the room and attributed to
  nobody — never guessed onto a member.
"""

from __future__ import annotations

import re
from typing import Any

#: The P5 settings block (deep-merged into every group's
#: settings like the other organs' blocks).  Both duties ship
#: OFF — the §28 precedent: a configured system changes no
#: group's behavior until the group turns it on.
P5_DEFAULTS: dict[str, Any] = {
    "topics_rename": False,     # rename auto-named forum
                                # topics from their first
                                # message (bot surface carries
                                # the is_name_implicit flag)
    "member_scopes": False,     # per-member Telegram command
                                # scopes for this room's admins
                                # (bot surface sets them via
                                # setMyCommands)
    "ephemeral": False,         # sensitive command replies
                                # (the §16 ephemeral class, e.g.
                                # /balance) delivered to the
                                # asker only in this room
}

#: Telegram's own tag law: 0–16 characters, no emoji.  The
#: empty tag CLEARS.  Devon additionally refuses markup-shaped
#: tags outright — a tag is rendered on presented surfaces.
TAG_MAX = 16

_TAG_BAD = re.compile(r"[@<>{}\[\]`*_~|#]")


def _is_emoji(ch: str) -> bool:
    """Emoji/pictograph ranges only — scripts like CJK or
    Cyrillic are ordinary letters and pass."""
    o = ord(ch)
    return (0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF
            or 0x2B00 <= o <= 0x2BFF or o == 0xFE0F)


def validate_tag(raw: str) -> tuple[bool, str, str]:
    """Judge a proposed member tag.  Returns (ok, cleaned,
    error): the cleaned tag ("" = clear) when ok, else the
    reason in plain words for the admin."""
    tag = " ".join(str(raw or "").split()).strip()
    if not tag:
        return True, "", ""
    if len(tag) > TAG_MAX:
        return False, "", (f"tags run to {TAG_MAX} characters "
                           f"at most — that one's "
                           f"{len(tag)}.")
    if any(_is_emoji(ch) for ch in tag):
        return False, "", "tags can't carry emoji."
    if _TAG_BAD.search(tag):
        return False, "", ("tags are plain words — no @handles, "
                           "links or markup characters.")
    return True, tag, ""


_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_LINK = re.compile(r"(https?://|www\.|t\.me/)\S*", re.IGNORECASE)


def derive_topic_name(text: str, *, limit: int = 48) -> str:
    """A proper name for an auto-named forum topic, derived
    from its first message: the first clause's words, cleaned
    of links and handles, capped for the topic-name field.
    "" when the message carries nothing nameable (a sticker,
    a bare link) — the duty then leaves the topic as it is
    rather than inventing a name."""
    body = _LINK.sub("", str(text or "").strip())
    first_line = body.splitlines()[0] if body else ""
    words = _WORD.findall(first_line)
    name = " ".join(words[:6]).strip()
    if len(name) < 3:
        return ""
    if len(name) > limit:
        name = name[:limit].rsplit(" ", 1)[0].strip() or name[:limit]
    return name[:1].upper() + name[1:]


#: Boost sources Telegram names (ChatBoostSource): a member's
#: own Premium, a gift code (4 boosts, attributed to the gifter
#: when claimed), a giveaway (attributed only when claimed).
BOOST_SOURCES = ("premium", "gift_code", "giveaway", "unknown")


def boost_source_kind(source: dict[str, Any] | None) -> str:
    """The source kind of a ChatBoostSource dict, by its
    ``source`` discriminator ("premium" | "gift_code" |
    "giveaway"); anything unrecognized is "unknown"."""
    kind = str((source or {}).get("source") or "").strip().lower()
    return kind if kind in BOOST_SOURCES else "unknown"
