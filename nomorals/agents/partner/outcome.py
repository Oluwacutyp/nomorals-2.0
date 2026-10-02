"""PresenceOutcome: what one inbound message produced."""

from __future__ import annotations

from dataclasses import dataclass, field
from ...partner.presence import Presence

@dataclass
class PresenceOutcome:
    """What one inbound message produced, presence decisions included.

    * ``parts`` non-empty — reply to send now.
    * ``parts`` empty + ``presence.delay_seconds`` — she's busy; the runtime
      schedules :meth:`PartnerBrain.deliver_reply` after the gap.
    * ``parts`` empty, no delay — read and left (low stakes) or a chat kind
      she stays quiet in.
    """

    parts: list[str] = field(default_factory=list)
    presence: Presence = field(default_factory=lambda: Presence(reply=True))
