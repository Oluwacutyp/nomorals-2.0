"""The partner runtime package (Wave H3 split of partner_runtime.py).

Public surface (all ``is``-identical to the historical
``nomorals.agents.partner_runtime`` names):

* :class:`PartnerBrain` — stateless-per-message cognition (.brain)
* :class:`PartnerRuntime` — the cross-platform loop (.runtime + mixins)
* :class:`PresenceOutcome` — one message's presence result (.outcome)
* ``_key_set`` / ``_direct_approve`` / ``_direct_deny`` — proposal helpers
  (.approvals), also re-exported by the ``partner_runtime`` facade.
"""

from __future__ import annotations

from .approvals import _direct_approve, _direct_deny, _key_set
from .brain import PartnerBrain
from .outcome import PresenceOutcome
from .runtime import PartnerRuntime

__all__ = ["PartnerBrain", "PartnerRuntime", "PresenceOutcome"]
