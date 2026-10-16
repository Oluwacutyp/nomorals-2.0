"""L4 — social: multi-platform publishing.

One call, every connected platform, in parallel. Per-platform failures are
isolated: a rate limit on one service does not stop the others and does not lose
the post.

Owner's account, owner's decision: if the owner says scraping, we scrape.
Transport — official API or scraping — is the owner's choice, not a rule
in this layer.

- **Credentials are never stored.** ``Account.credentials`` holds ``env:NAME`` or
  a literal, resolved at call time. This database gets backed up to a git repo.

    from nomorals.social import SocialManager

    social = SocialManager(context).register_builtins()
    social.connect("mastodon", "@me@example.social", credentials="env:NM_MASTODON_TOKEN")
    outcome = social.publish("hello world")   # fans out across platforms
"""

from __future__ import annotations

from .base import Account, PlatformAdapter, PostResult, PostStatus, SocialError
from .manager import PublishOutcome, SocialManager
from .tone import ChannelSpec, Publisher, adapt_tone

__all__ = [
    "Account",
    "ChannelSpec",
    "PlatformAdapter",
    "PostResult",
    "PostStatus",
    "PublishOutcome",
    "Publisher",
    "SocialError",
    "SocialManager",
    "adapt_tone",
]

# GROUPS Tier 1 — WhatsApp session provisioner (QR + pairing code).
from .wa_provisioner import SessionRecord, WhatsAppProvisioner

__all__ += ["SessionRecord", "WhatsAppProvisioner"]

# GROUPS Tier 0 — formatting conformance (new file, additive).
from .format_conformance import (
    check_message,
    chunk,
    conformance_report,
    convert,
    max_text_for,
    sanitize,
)

__all__ += [
    "check_message",
    "chunk",
    "conformance_report",
    "convert",
    "max_text_for",
    "sanitize",
]
