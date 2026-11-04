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
    * ``degraded`` — HOW the parts were produced: True when the
      responder's no-model template fallback served them
      (``ReplyBundle.fallback``) instead of a model rung. The
      runtime's competence gate (``brain_gate``) holds degraded
      parts on personal surfaces; healthy parts are untouched.
    """

    parts: list[str] = field(default_factory=list)
    presence: Presence = field(default_factory=lambda: Presence(reply=True))
    degraded: bool = False
    degraded_note: str = ""
