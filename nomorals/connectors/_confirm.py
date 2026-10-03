"""Shared explicit-confirmation helper for consequential connector actions.

Some connector calls move value or information out of the owner's accounts
(sending mail, charging a card, placing a trade). Those calls never run on
an agent's bare say-so. This module enforces one rule everywhere:

* ``confirmed=True`` — the caller attests the owner already approved the
  exact payload. Proceed immediately.
* ``db`` given (and not confirmed) — open a human checkpoint carrying the
  exact payload; interactive TTYs wait for the owner, otherwise
  :class:`HumanCheckpointPending` pauses the flow cleanly. The connector's
  ``resume_checkpoint`` completes the action when the checkpoint resolves.
* neither — fail fast with a message naming the exact confirmation needed.

No silent success, no implied consent.
"""

from __future__ import annotations

from typing import Any

from .base import Connector, ConnectorError
from .checkpoints import CheckpointKind, CheckpointState

__all__ = ["confirm_or_checkpoint"]


def confirm_or_checkpoint(
    connector: Connector,
    *,
    confirmed: bool,
    db: Any,
    context: Any,
    stage: str,
    title: str,
    instructions: str,
    resume_state: dict[str, Any] | None = None,
) -> bool:
    """Gate a consequential action behind explicit owner confirmation.

    Returns True when the action may proceed now. Raises
    :class:`HumanCheckpointPending` (non-interactive) when the decision is
    parked on a human checkpoint, or :class:`ConnectorError` when there is
    no confirmation and no way to ask for one.
    """
    if confirmed:
        return True
    if db is None:
        raise ConnectorError(
            f"{title}: refusing to run without explicit owner confirmation — "
            "the owner must approve the exact payload first, then call again "
            "with confirmed=True"
        )
    state = dict(resume_state or {})
    state["stage"] = stage
    checkpoint = connector.request_human(
        CheckpointKind.MANUAL_STEP,
        title,
        instructions,
        db=db,
        context=context,
        resume_state=state,
    )
    # request_human either raised HumanCheckpointPending (non-interactive)
    # or returned an already-resolved checkpoint (interactive TTY).
    if checkpoint.state != CheckpointState.RESOLVED:
        raise ConnectorError(
            f"confirmation checkpoint {checkpoint.id} is "
            f"{checkpoint.state.value}, not resolved — the owner must "
            "approve first"
        )
    return True
