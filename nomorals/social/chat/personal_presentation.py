"""Personal-presentation gating for social sends (code-enforced policy helper).

Devon never acts like an AI outside the owner's DM / owner-originated
chats. The LIVE enforcement of that rule is:

* :mod:`nomorals.partner.gating` — prompt-level restriction
  (``gate_decision`` / ``is_restricted``), consulted by the partner
  brain and runtime on every inbound message;
* :mod:`nomorals.partner.social_gate` — tool-call grants
  (``grant_for`` / ``check_tool_call``), consulted by the agent loop
  before a tool may act in a non-owner chat.

This module is a TESTED POLICY HELPER for send-time checks — it is NOT
wired into the send path and is not "the enforcement point every social
send funnels through". It exists for callers that need an explicit
policy decision about presenting as Devon (or as the owner) to a
non-owner, e.g. future proactive-send flows:

* **Identity.** A send that would present as Devon (or as the owner)
  to a non-owner must carry ``SOCIAL_DM`` (1:1) or ``SOCIAL_BULK``
  (multi-recipient / broadcast) — both are in
  :data:`Capability.CONFIRMABLE`, so the policy layer demands an
  explicit owner confirmation token, not just a grant.
* **Owner-only.** Non-owner actors can never gain SOCIAL_DM/SOCIAL_BULK
  through this gate, whatever their role preset says: the gate checks
  ``is_owner`` first, before the policy grant.
* **Confirmation.** The caller mints a token with
  ``policy.issue_confirmation(capability)`` after the owner approves
  this exact send, and passes it as ``confirmation``. Without it the
  decision comes back ``allowed=False, needs_confirmation=True`` —
  the caller prompts, never sends. ``explicit_override=True`` is the
  "post it" path (explicit owner instruction bypasses the token gate;
  deny rules and biometric rules still apply).

Returns a :class:`PolicyDecision` from the real policy layer
(:meth:`Policy.check`), so the audit trail, gradient, and biometric
rules all apply uniformly. ``raise_if_denied`` on the result is the
one-line enforcement at a send site.
"""

from __future__ import annotations

from ...core.policy import Capability, Policy, PolicyDecision

__all__ = ["check_personal_presentation", "capability_for_send"]


def capability_for_send(*, recipients: int, is_group: bool = False) -> str:
    """SOCIAL_DM for 1:1, SOCIAL_BULK for groups / multi-recipient."""
    if recipients > 1 or is_group:
        return Capability.SOCIAL_BULK
    return Capability.SOCIAL_DM


def check_personal_presentation(
    policy: Policy | None,
    *,
    actor: str,
    is_owner: bool,
    recipients: int = 1,
    is_group: bool = False,
    confirmation: str | None = None,
    explicit_override: bool = False,
) -> PolicyDecision:
    """Gate one social send through the policy layer.

    ``policy=None`` fails closed (deny). Non-owner actors are denied
    outright — personal presentation is owner-only. Owner sends without
    a ``confirmation`` token come back ``needs_confirmation=True``:
    the caller prompts for confirmation, mints the token, and re-calls
    — the gate never auto-approves a confirmable capability.
    """
    capability = capability_for_send(recipients=recipients, is_group=is_group)
    if not is_owner:
        return PolicyDecision(
            allowed=False, capability=capability, actor=actor,
            reason="personal presentation is owner-only: non-owner actors "
                   "cannot send as Devon or as the owner.")
    if policy is None:
        return PolicyDecision(
            allowed=False, capability=capability, actor=actor,
            reason="no policy attached — failing closed.")
    return policy.check(capability, actor=actor, confirmation=confirmation,
                        explicit_override=explicit_override)
