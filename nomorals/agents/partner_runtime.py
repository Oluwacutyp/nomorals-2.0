"""The partner runtime: the brain, and the loop that keeps it alive.

Two objects:

* :class:`PartnerBrain` — stateless-per-message cognition. For one inbound
  message it: advances the mood clock, detects emotional signals, recalls
  shared memory, selects gated background knowledge, calls the model, guards
  the output, and persists everything (messages, training pairs).
* :class:`PartnerRuntime` — the loop. It owns the :class:`ChatGateway`, runs
  every platform at once, guarantees **per-chat FIFO ordering** while letting
  different chats be processed **in parallel** (bounded by
  ``partner.max_parallel_chats``), adds typing indicators, splits replies
  into short human-sized sends, and spawns the curator sub-agent on segments.

Conversation rules per chat kind:

* **DM (the partner / owner)** — always answered.
* **DM (anyone else)** — answered in character, but the context tells her this
  is not the partner, so the intimacy stays where it belongs.
* **Group** — she listens; she only speaks when mentioned, when someone
  replies to her, or when the partner is talking there.
* **Channel** — read-only for her; posting to channels is the autonomy
  agent's job (deliberately, on its own schedule).

Compatibility facade (Wave H3): the implementation now lives in the
:mod:`nomorals.agents.partner` subpackage (``brain.py``,
``runtime.py`` + ``runtime_*`` mixins, ``outcome.py``, ``approvals.py``).
This module re-exports the exact same objects, so every existing
``from nomorals.agents.partner_runtime import ...`` keeps working.
"""

from __future__ import annotations

from .partner import (
    PartnerBrain,
    PartnerRuntime,
    PresenceOutcome,
    _direct_approve,
    _direct_deny,
    _key_set,
)

__all__ = ["PartnerBrain", "PartnerRuntime", "PresenceOutcome"]
