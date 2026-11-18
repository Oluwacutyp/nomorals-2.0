"""Discord tickets filed as GitHub issues — the ticket relay.

Ported contract from storytold/discord-github-ticket-relay (MIT):
the issue title/body shape, the submission marker, @-mention
neutralisation, duplicate protection. Devon-native trigger: the
/ticket verb in a channel Devon serves, plus the ``ticket_relay``
brain tool. The loop closes by polling (upstream's webhook is
optional; Devon has no public webhook endpoint): the hourly sync
tick pulls issue state and comments back to Discord.
"""
from __future__ import annotations

from . import content, jobs, relay, store, surface

__all__ = ["content", "jobs", "relay", "store", "surface"]
