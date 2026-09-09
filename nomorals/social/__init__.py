"""L4 — social: multi-platform publishing on official APIs.

One call, every connected platform, in parallel. Per-platform failures are
isolated: a rate limit on one service does not stop the others and does not lose
the post.

Two rules this layer does not bend:

- **Official APIs only.** Scraping to automate violates terms of service, breaks
  on every markup change, and gets accounts banned.
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

__all__ = [
    "Account",
    "PlatformAdapter",
    "PostResult",
    "PostStatus",
    "PublishOutcome",
    "SocialError",
    "SocialManager",
]
