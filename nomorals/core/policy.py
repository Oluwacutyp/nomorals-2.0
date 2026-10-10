"""Capability policy.

"You own the model" does not mean "nothing is gated". An autonomous agent tree
with 64 sub-agents and shell access will eventually do something you did not
intend — not out of malice, out of a misparsed glob. Gating destructive actions
costs legitimate workflows nothing and makes the system safe to leave running
unattended, which is the entire point of an autonomous system.

Model
-----
* A **capability** is a dotted string: ``fs.write``, ``exec.shell``, ``net.out``,
  ``social.post``, ``db.write``, ``train.gpu`` …
* An **actor** (agent, mission, tool call) holds a :class:`CapabilitySet`.
* A tool declares the capabilities it needs. The policy intersects the actor's
  grant with what the tool asks for and returns a :class:`PolicyDecision`.
* Sub-agents inherit ``parent ∩ role_requirement`` — privilege narrows down the
  tree, never widens.
* **Confirmable** capabilities additionally require a one-time confirmation token
  issued by the operator out of band. This is the gate on ``rm -rf``, ``DROP
  TABLE``, bulk DM, and bulk follow.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from .logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "AUDIT_BIOMETRIC",
    "AUDIT_DENY",
    "AUDIT_GRANT",
    "AUDIT_OBSERVE",
    "AUDIT_PROPOSE",
    "AUDIT_REVOKE",
    "GRADIENT_ACT_SILENT",
    "GRADIENT_ACT_WITH_APPROVAL",
    "GRADIENT_OBSERVE",
    "GRADIENT_PROPOSE",
    "Capability",
    "CapabilitySet",
    "MemoryPolicyStore",
    "PermissionGradient",
    "Policy",
    "PolicyDecision",
    "PolicyStore",
    "TimedGrant",
    "approve_with_biometric",
    "is_explicit_instruction",
    "narrow_grant",
]


class Capability:
    """The capability namespace. Strings, not an enum, so tools can extend it."""

    # Filesystem
    FS_READ = "fs.read"
    FS_WRITE = "fs.write"
    FS_DELETE = "fs.delete"
    # Shell / code execution
    EXEC_SHELL = "exec.shell"
    EXEC_CODE = "exec.code"
    EXEC_INSTALL = "exec.install"
    # Network
    NET_OUT = "net.out"
    NET_DOWNLOAD = "net.download"
    NET_BROWSER = "net.browser"
    # Model plane
    MODEL_CALL = "model.call"
    MODEL_DOWNLOAD = "model.download"
    TRAIN_RUN = "train.run"
    TRAIN_GPU = "train.gpu"
    # Memory / data
    MEM_READ = "mem.read"
    MEM_WRITE = "mem.write"
    DB_READ = "db.read"
    DB_WRITE = "db.write"
    DB_ADMIN = "db.admin"
    # Agents
    AGENT_SPAWN = "agent.spawn"
    MISSION_START = "mission.start"
    # Media
    MEDIA = "media.gen"
    # Social
    SOCIAL_POST = "social.post"
    SOCIAL_READ = "social.read"
    SOCIAL_DM = "social.dm"
    SOCIAL_BULK = "social.bulk"
    # System
    SYS_CONFIG = "sys.config"
    SYS_BACKUP = "sys.backup"
    SYS_SHUTDOWN = "sys.shutdown"

    ALL: tuple[str, ...] = (
        FS_READ, FS_WRITE, FS_DELETE,
        EXEC_SHELL, EXEC_CODE, EXEC_INSTALL,
        NET_OUT, NET_DOWNLOAD, NET_BROWSER,
        MODEL_CALL, MODEL_DOWNLOAD, TRAIN_RUN, TRAIN_GPU,
        MEM_READ, MEM_WRITE, DB_READ, DB_WRITE, DB_ADMIN,
        AGENT_SPAWN, MISSION_START,
        SOCIAL_POST, SOCIAL_READ, SOCIAL_DM, SOCIAL_BULK,
        SYS_CONFIG, SYS_BACKUP, SYS_SHUTDOWN,
        MEDIA,
    )

    #: Capabilities that require an explicit operator confirmation token.
    CONFIRMABLE: frozenset[str] = frozenset(
        {FS_DELETE, EXEC_INSTALL, DB_ADMIN, SOCIAL_DM, SOCIAL_BULK, SYS_SHUTDOWN}
    )

    #: Capabilities that require *biometric* (fingerprint) approval on top of
    #: the confirmation token. A subset of the irreversible ones: the token
    #: for these must be minted through the fingerprint prompt
    #: (:func:`approve_with_biometric`), never through the text flow.
    BIOMETRIC: frozenset[str] = frozenset({FS_DELETE, DB_ADMIN, SYS_SHUTDOWN})


#: Preset grants. Roles pick one; operators can widen or narrow it.
ROLE_PRESETS: dict[str, tuple[str, ...]] = {
    "research": (
        Capability.NET_OUT,
        Capability.NET_BROWSER,
        Capability.NET_DOWNLOAD,
        Capability.FS_READ,
        Capability.FS_WRITE,
        Capability.MEM_READ,
        Capability.MEM_WRITE,
        Capability.MODEL_CALL,
        Capability.DB_READ,
        Capability.DB_WRITE,
    ),
    "coding": (
        Capability.FS_READ,
        Capability.FS_WRITE,
        Capability.EXEC_CODE,
        Capability.EXEC_SHELL,
        Capability.MEM_READ,
        Capability.MEM_WRITE,
        Capability.MODEL_CALL,
        Capability.DB_READ,
        Capability.DB_WRITE,
    ),
    "vision": (
        Capability.FS_READ,
        Capability.NET_DOWNLOAD,
        Capability.MODEL_CALL,
        Capability.MEM_WRITE,
    ),
    "data_collection": (
        Capability.NET_OUT,
        Capability.NET_DOWNLOAD,
        Capability.FS_READ,
        Capability.FS_WRITE,
        Capability.MODEL_CALL,
        Capability.DB_READ,
        Capability.DB_WRITE,
    ),
    "training": (
        Capability.FS_READ,
        Capability.FS_WRITE,
        Capability.EXEC_SHELL,
        Capability.TRAIN_RUN,
        Capability.TRAIN_GPU,
        Capability.MODEL_DOWNLOAD,
        Capability.DB_READ,
        Capability.DB_WRITE,
    ),
    "social": (
        Capability.SOCIAL_POST,
        Capability.SOCIAL_READ,
        Capability.NET_OUT,
        Capability.FS_READ,
        Capability.MEM_READ,
        Capability.MEM_WRITE,
        Capability.DB_READ,
        Capability.DB_WRITE,
    ),
    "memory": (
        Capability.MEM_READ,
        Capability.MEM_WRITE,
        Capability.DB_READ,
        Capability.DB_WRITE,
        Capability.MODEL_CALL,
    ),
    "execution": (
        Capability.FS_READ,
        Capability.FS_WRITE,
        Capability.EXEC_CODE,
        Capability.EXEC_SHELL,
        Capability.NET_OUT,
        Capability.DB_READ,
        Capability.DB_WRITE,
    ),
    "orchestrator": tuple(Capability.ALL),
    # Local operator. Unrestricted by construction ("*"), not by enumeration, so
    # future extension capabilities are covered too. Policy.grant_for_role pins
    # this role explicitly: a default_grant ceiling never narrows it.
    "owner": ("*",),
    "critic": (Capability.MEM_READ, Capability.DB_READ, Capability.MODEL_CALL),
    "readonly": (Capability.FS_READ, Capability.MEM_READ, Capability.DB_READ),
}


@dataclass(frozen=True)
class CapabilitySet:
    """An immutable set of capability patterns (``fs.*`` and ``*`` supported)."""

    patterns: frozenset[str] = frozenset()

    @classmethod
    def all(cls) -> CapabilitySet:
        return cls(frozenset({"*"}))

    @classmethod
    def none(cls) -> CapabilitySet:
        return cls(frozenset())

    @classmethod
    def of(cls, *caps: str) -> CapabilitySet:
        return cls(frozenset(caps))

    @classmethod
    def role(cls, role: str) -> CapabilitySet:
        if role not in ROLE_PRESETS:
            raise KeyError(f"unknown role preset {role!r}; known: {sorted(ROLE_PRESETS)}")
        return cls(frozenset(ROLE_PRESETS[role]))

    def grants(self, capability: str) -> bool:
        if "*" in self.patterns or capability in self.patterns:
            return True
        return any(fnmatch.fnmatchcase(capability, p) for p in self.patterns)

    def union(self, other: CapabilitySet) -> CapabilitySet:
        return CapabilitySet(self.patterns | other.patterns)

    def intersect(self, other: CapabilitySet) -> CapabilitySet:
        """Narrow to capabilities both sides allow.

        Wildcards are handled conservatively: if either side is unrestricted the
        result is the other side; otherwise the literal intersection.
        """
        if "*" in self.patterns:
            return other
        if "*" in other.patterns:
            return self
        expanded_self = _expand(self.patterns)
        expanded_other = _expand(other.patterns)
        return CapabilitySet(frozenset(expanded_self & expanded_other))

    def minus(self, *caps: str) -> CapabilitySet:
        expanded = _expand(self.patterns)
        for cap in caps:
            expanded = {c for c in expanded if not fnmatch.fnmatchcase(c, cap)}
        return CapabilitySet(frozenset(expanded))

    def as_list(self) -> list[str]:
        if "*" in self.patterns:
            return ["*"]
        return sorted(_expand(self.patterns))

    def __len__(self) -> int:
        return len(self.patterns)

    def __contains__(self, capability: str) -> bool:
        return self.grants(capability)


def _expand(patterns: Iterable[str]) -> set[str]:
    """Resolve wildcard patterns against the known capability namespace."""
    out: set[str] = set()
    for pattern in patterns:
        if pattern == "*":
            out.update(Capability.ALL)
        elif "*" in pattern or "?" in pattern:
            out.update(c for c in Capability.ALL if fnmatch.fnmatchcase(c, pattern))
        else:
            out.add(pattern)
    return out


AUDIT_ALLOW = "allow"
AUDIT_DENY = "deny"
AUDIT_CONFIRM = "confirm"
AUDIT_BIOMETRIC = "biometric"
AUDIT_OBSERVE = "observe"
AUDIT_PROPOSE = "propose"
AUDIT_GRANT = "grant"
AUDIT_REVOKE = "revoke"


#: Permission-gradient levels (Dots pattern: read-only-proactive →
#: engaged-active). The gradient governs *autonomous* action; explicit
#: owner instructions bypass it (see :func:`is_explicit_instruction`).
GRADIENT_OBSERVE = "observe"
GRADIENT_PROPOSE = "propose"
GRADIENT_ACT_WITH_APPROVAL = "act_with_approval"
GRADIENT_ACT_SILENT = "act_silent"


@dataclass
class PolicyDecision:
    """Result of a policy check."""

    allowed: bool
    reason: str = ""
    capability: str = ""
    actor: str = ""
    needs_confirmation: bool = False
    needs_biometric: bool = False
    needs_proposal: bool = False
    proposal_id: str = ""
    gradient: str = ""
    explicit_override: bool = False
    audit_id: str = ""

    def __bool__(self) -> bool:
        return self.allowed

    def raise_if_denied(self) -> None:
        if not self.allowed:
            from .errors import CapabilityDenied

            raise CapabilityDenied(
                self.reason or "capability denied",
                capability=self.capability,
                actor=self.actor,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "capability": self.capability,
            "actor": self.actor,
            "needs_confirmation": self.needs_confirmation,
            "needs_biometric": self.needs_biometric,
            "needs_proposal": self.needs_proposal,
            "proposal_id": self.proposal_id,
            "gradient": self.gradient,
            "explicit_override": self.explicit_override,
            "audit_id": self.audit_id,
        }


@dataclass
class _Rule:
    capability: str
    effect: str  # allow | deny | confirm | biometric
    note: str = ""
    priority: int = 0
    # Optional condition evaluated against the check's ``context`` dict.
    # The rule only applies when ``when(context)`` is truthy; a condition
    # that raises fails CLOSED (the check is denied). Context is no longer
    # just audit metadata — it is how rules get resource-bound, e.g.
    # ``allow("fs.write", when=lambda ctx: ctx["path"].startswith(ws))``.
    when: Callable[[dict[str, Any]], bool] | None = None


class Policy:
    """Evaluates capability requests against rules, grants, and confirmations.

    Evaluation order:
      1. explicit ``deny`` rules
      2. ``biometric`` / ``confirm`` rules → require a valid confirmation token
         (biometric rules additionally require the token to be minted through
         the fingerprint prompt)
      3. explicit ``allow`` rules
      4. the actor's :class:`CapabilitySet` (or :class:`TimedGrant`)
      5. default deny

    Rules may carry a ``when`` condition evaluated against the check's
    ``context`` — the rule only applies when the condition holds, which is
    how rules get resource-bound (e.g. ``fs.write`` allowed only under the
    workspace directory). A condition that raises fails closed: the check
    is denied.
    """

    def __init__(
        self,
        *,
        default_grant: CapabilitySet | None = None,
        enforce: bool = True,
        confirmation_ttl: float = 300.0,
        clock: Any = None,
        progression: Callable[[], set[str]] | None = None,
    ) -> None:
        """Args:
            progression: optional zero-arg callable returning extra granted
                capability strings for the actor (e.g. game-achievement
                unlocks). Consulted alongside the actor's CapabilitySet at
                the grant step. Never raises — failures degrade to no extra
                grants. Kept as a callable so policy.py stays decoupled from
                the games DB.
        """
        self.default_grant = default_grant if default_grant is not None else CapabilitySet.none()
        self.enforce = enforce
        self.confirmation_ttl = confirmation_ttl
        self.progression = progression
        self._rules: list[_Rule] = []
        # token -> (expiry_ts, capability it was minted for)
        self._confirmations: dict[str, tuple[float, str]] = {}
        self._audit: list[dict[str, Any]] = []
        self._audit_limit = 2000
        self._lock = threading.RLock()
        self._clock = clock or _DefaultClock()
        self._counts = {
            "allow": 0,
            "deny": 0,
            "confirm": 0,
            "biometric": 0,
            "observe": 0,
            "propose": 0,
        }

    # -- rule management -----------------------------------------------------
    def allow(self, capability: str, *, note: str = "", priority: int = 10,
              when: Callable[[dict[str, Any]], bool] | None = None) -> Policy:
        """Allow ``capability``; with ``when``, only when the condition holds
        for the check's context (resource-bound allow).

        Resource-binding pattern — allow ``fs.write`` ONLY under the
        workspace directory::

            policy.deny("fs.write", note="writes stay in the workspace")
            policy.allow("fs.write", priority=200, note="workspace writes",
                         when=lambda ctx: str(ctx.get("path", "")).startswith(ws))

        The conditional allow outranks the deny (priority); when its
        condition is false the deny fires and the write is refused.
        """
        return self._add(_Rule(capability, "allow", note, priority, when))

    def deny(self, capability: str, *, note: str = "", priority: int = 100,
             when: Callable[[dict[str, Any]], bool] | None = None) -> Policy:
        """Deny ``capability``; with ``when``, only when the condition holds
        for the check's context."""
        return self._add(_Rule(capability, "deny", note, priority, when))

    def confirm(self, capability: str, *, note: str = "", priority: int = 50,
                when: Callable[[dict[str, Any]], bool] | None = None) -> Policy:
        return self._add(_Rule(capability, "confirm", note, priority, when))

    def biometric(self, capability: str, *, note: str = "", priority: int = 60,
                  when: Callable[[dict[str, Any]], bool] | None = None) -> Policy:
        """Gate on fingerprint approval: the confirmation token for this
        capability must be minted through :func:`approve_with_biometric`."""
        return self._add(_Rule(capability, "biometric", note, priority, when))

    def list_rules(self) -> list[dict[str, Any]]:
        """Introspection: every rule with its effect, priority, and whether
        it carries a condition. Powers ``nm policy rules``."""
        with self._lock:
            return [
                {
                    "capability": r.capability,
                    "effect": r.effect,
                    "note": r.note,
                    "priority": r.priority,
                    "conditional": r.when is not None,
                }
                for r in self._rules
            ]

    def _add(self, rule: _Rule) -> Policy:
        with self._lock:
            self._rules.append(rule)
            self._rules.sort(key=lambda r: -r.priority)
        return self

    def clear_rules(self) -> None:
        with self._lock:
            self._rules.clear()

    # -- confirmations -------------------------------------------------------
    def issue_confirmation(self, capability: str, *, ttl: float | None = None) -> str:
        """Mint a single-use token authorising exactly one confirmable action.

        The token is bound to the capability it was minted for, so a token issued
        for ``fs.delete`` cannot be replayed to authorise ``social.bulk``.
        """
        from .ids import new_short_id

        token = new_short_id("cfm_") + hashlib.sha256(
            f"{capability}:{self._clock.now()}".encode()
        ).hexdigest()[:16]
        with self._lock:
            self._confirmations[token] = (
                self._clock.now() + (ttl or self.confirmation_ttl),
                capability,
            )
        return token

    def _consume_confirmation(self, token: str, capability: str) -> bool:
        """Validate and consume a token. Single-use and capability-bound."""
        with self._lock:
            entry = self._confirmations.pop(token, None)
            if entry is None:
                return False
            expiry, bound_capability = entry
            if expiry < self._clock.now():
                return False
            return bound_capability == capability

    def pending_confirmations(self) -> int:
        with self._lock:
            now = self._clock.now()
            self._confirmations = {
                t: e for t, e in self._confirmations.items() if e[0] >= now
            }
            return len(self._confirmations)

    def requires_biometric(self, capability: str) -> bool:
        """True when this capability needs fingerprint approval.

        Mirrors :meth:`check`'s rule evaluation without recording an audit
        entry: the first matching rule decides (``biometric`` → True,
        ``deny`` → False since the action is refused outright); otherwise
        the :attr:`Capability.BIOMETRIC` set applies. Conditional rules are
        invisible here (no context to evaluate them against) — the real
        :meth:`check` with context is authoritative. Never raises.
        """
        try:
            with self._lock:
                for rule in self._rules:
                    if not fnmatch.fnmatchcase(capability, rule.capability):
                        continue
                    if rule.effect == "deny":
                        return False
                    if rule.effect == "biometric":
                        return True
                    # confirm/allow: rule settled the level; the BIOMETRIC
                    # set still applies on top, same as in check().
                    break
            return capability in Capability.BIOMETRIC
        except Exception:  # noqa: BLE001 - fail closed on evaluation errors
            return True

    def _progression_grants(self, capability: str) -> bool:
        """True when the progression callable grants this capability.

        Additive-only grant source (game achievement unlocks). Never raises:
        a failing callable degrades to no extra grants.
        """
        prog = self.progression
        if prog is None:
            return False
        try:
            granted = prog()
        except Exception:  # noqa: BLE001 - fail closed on callable errors
            _log.debug("progression callable failed", exc_info=True)
            return False
        try:
            return capability in granted
        except Exception:  # noqa: BLE001
            return False

    # -- evaluation ----------------------------------------------------------
    def check(
        self,
        capability: str,
        *,
        actor: str = "",
        grant: CapabilitySet | None = None,
        confirmation: str | None = None,
        context: dict[str, Any] | None = None,
        explicit_override: bool = False,
        audit_kind: str | None = None,
    ) -> PolicyDecision:
        """Evaluate whether ``actor`` may exercise ``capability``.

        ``explicit_override``: an explicit owner instruction ("post it",
        "send it", "do it now") jumps straight to execution — the
        confirmation-token gate is bypassed. Deny rules and grants still
        apply, and biometric (fingerprint) approval remains mandatory.
        ``audit_kind``: optional audit-kind override for the allow record
        (used by the permission gradient's observe level).
        """
        effective = grant if grant is not None else self.default_grant
        reason = ""
        needs_confirm = False
        needs_biometric = False
        ctx = dict(context or {})
        if explicit_override:
            ctx["explicit_override"] = True

        with self._lock:
            for rule in self._rules:
                if not fnmatch.fnmatchcase(capability, rule.capability):
                    continue
                if rule.when is not None:
                    try:
                        holds = bool(rule.when(ctx))
                    except Exception:  # noqa: BLE001 - a broken condition fails CLOSED
                        _log.warning(
                            "policy rule condition raised for %s; denying",
                            rule.capability, exc_info=True)
                        holds = False
                        broken = True
                    else:
                        broken = False
                    if broken:
                        decision = PolicyDecision(
                            allowed=False,
                            reason=(f"policy rule condition failed for "
                                    f"{rule.capability!r}; failing closed"),
                            capability=capability,
                            actor=actor,
                            explicit_override=explicit_override,
                        )
                        self._record(AUDIT_DENY, decision, ctx)
                        return decision
                    if not holds:
                        continue  # condition false: rule does not apply
                if rule.effect == "deny":
                    decision = PolicyDecision(
                        allowed=False,
                        reason=rule.note or f"denied by policy rule: {rule.capability}",
                        capability=capability,
                        actor=actor,
                        explicit_override=explicit_override,
                    )
                    self._record(AUDIT_DENY, decision, ctx)
                    return decision
                if rule.effect == "biometric":
                    needs_confirm = True
                    needs_biometric = True
                    reason = rule.note or f"requires biometric approval: {rule.capability}"
                    break
                if rule.effect == "confirm":
                    needs_confirm = True
                    reason = rule.note or f"requires confirmation: {rule.capability}"
                    break
                if rule.effect == "allow":
                    break

        granted = effective.grants(capability)
        if not granted:
            # Step 4b: progression unlocks (achievement grants) — additive
            # only, consulted alongside the actor's CapabilitySet.
            granted = self._progression_grants(capability)
        confirmable = needs_confirm or capability in Capability.CONFIRMABLE
        biometric_required = needs_biometric or capability in Capability.BIOMETRIC

        if not self.enforce:
            decision = PolicyDecision(
                allowed=True,
                reason="policy enforcement disabled",
                capability=capability,
                actor=actor,
                explicit_override=explicit_override,
            )
            self._record(AUDIT_ALLOW, decision, ctx)
            return decision

        if not granted:
            decision = PolicyDecision(
                allowed=False,
                reason=f"actor {actor or 'anonymous'!r} lacks capability {capability!r}",
                capability=capability,
                actor=actor,
                explicit_override=explicit_override,
            )
            self._record(AUDIT_DENY, decision, ctx)
            return decision

        # Explicit owner instructions bypass the confirmation-token gate
        # (execute without reconfirming), but biometric fingerprint approval
        # stays structural.
        skip_confirmation = explicit_override and not biometric_required
        if confirmable and not skip_confirmation and (
            not confirmation or not self._consume_confirmation(confirmation, capability)
        ):
            decision = PolicyDecision(
                allowed=False,
                reason=reason
                or f"capability {capability!r} requires an operator confirmation token",
                capability=capability,
                actor=actor,
                needs_confirmation=True,
                needs_biometric=biometric_required,
                explicit_override=explicit_override,
            )
            self._record(
                AUDIT_BIOMETRIC if biometric_required else AUDIT_CONFIRM,
                decision,
                ctx,
            )
            return decision

        decision = PolicyDecision(
            allowed=True,
            reason="granted (explicit owner instruction)"
            if explicit_override
            else "granted",
            capability=capability,
            actor=actor,
            explicit_override=explicit_override,
        )
        self._record(audit_kind or AUDIT_ALLOW, decision, ctx)
        return decision

    def require(
        self,
        capability: str,
        *,
        actor: str = "",
        grant: CapabilitySet | None = None,
        confirmation: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """Like :meth:`check` but raises :class:`CapabilityDenied` on denial."""
        decision = self.check(
            capability, actor=actor, grant=grant, confirmation=confirmation, context=context
        )
        decision.raise_if_denied()
        return decision

    # -- audit ---------------------------------------------------------------
    def _record(self, kind: str, decision: PolicyDecision, context: dict[str, Any] | None) -> None:
        from .ids import new_short_id

        entry = {
            "audit_id": new_short_id("aud_"),
            "ts": self._clock.now(),
            "kind": kind,
            "capability": decision.capability,
            "actor": decision.actor,
            "reason": decision.reason,
            "context": dict(context or {}),
        }
        decision.audit_id = entry["audit_id"]
        with self._lock:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            self._audit.append(entry)
            if len(self._audit) > self._audit_limit:
                del self._audit[: len(self._audit) - self._audit_limit]

    def audit_log(self, limit: int = 100, kind: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._audit)
        if kind:
            items = [i for i in items if i["kind"] == kind]
        return items[-limit:]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "rules": len(self._rules),
                "counts": dict(self._counts),
                "enforce": self.enforce,
                "audit_entries": len(self._audit),
            }

    def verify_decision(self, decision: PolicyDecision) -> bool:
        """Check a decision against the tamper-evident audit trail.

        Every :meth:`check` records its verdict *before* returning, so a
        decision whose ``allowed`` bit disagrees with its audit entry was
        tampered with after evaluation (a flipped grant).  Returns True
        when the decision is consistent with the audit record.
        """
        if not decision.audit_id:
            return False
        with self._lock:
            entry = next(
                (e for e in self._audit if e.get("audit_id") == decision.audit_id),
                None,
            )
        if entry is None:
            return False
        kind = entry.get("kind")
        if kind == AUDIT_ALLOW:
            return bool(decision.allowed)
        if kind == AUDIT_DENY:
            return not decision.allowed
        if kind == AUDIT_OBSERVE:
            return bool(decision.allowed)
        if kind == AUDIT_PROPOSE:
            # A proposal verdict: denied pending owner approval, carrying
            # the proposal id. A flipped bit would claim a grant that was
            # never approved.
            return (not decision.allowed) and bool(decision.proposal_id)
        # AUDIT_CONFIRM / AUDIT_BIOMETRIC: the verdict was "denied pending
        # confirmation"; a flipped bit would claim a grant that was never
        # confirmed.
        if kind in (AUDIT_CONFIRM, AUDIT_BIOMETRIC):
            return not decision.allowed or decision.needs_confirmation
        return False

    # -- helpers -------------------------------------------------------------
    def grant_for_role(self, role: str) -> CapabilitySet:
        """Grant capabilities for a role preset, honoring ``default_grant`` as a ceiling.

        Semantics (Wave H3 audit — chosen deliberately):

        * Unknown roles raise :exc:`KeyError` via :meth:`CapabilitySet.role`,
          as before.
        * The owner role — any preset containing the ``"*"`` wildcard — is the
          operator's own unrestricted grant and is returned unchanged. A
          ceiling exists to narrow *delegated* roles, never to clip the
          operator's full power. The pin must be explicit: plain
          ``preset.intersect(default_grant)`` would hand a wildcard left side
          straight back as ``default_grant`` (see
          :meth:`CapabilitySet.intersect`), silently demoting the owner.
        * With no ceiling configured (``default_grant`` empty/unset, the
          default), the preset passes through unchanged. This preserves the
          method's historical observable behavior — ``default_grant`` was
          previously dead code here (it was unioned with
          ``CapabilitySet.all()``, whose ``"*"`` made ``intersect`` return the
          role preset untouched) — so nothing silently tightens.
        * With a non-empty ceiling configured, the result is
          ``preset ∩ default_grant``: the ceiling actually narrows the role
          grant, which is what the constructor parameter always promised.
        """
        preset = CapabilitySet.role(role)
        if "*" in preset.patterns:
            return preset
        if not self.default_grant.patterns:
            return preset
        return preset.intersect(self.default_grant)

    # -- timed grants ----------------------------------------------------------
    def issue_grant(
        self,
        *patterns: str,
        ttl_s: float,
        actor: str = "",
        note: str = "",
    ) -> "TimedGrant":
        """Mint a time-bounded capability grant.

        The grant is usable as ``grant=`` in :meth:`check` anywhere a
        :class:`CapabilitySet` goes. It stops granting the moment it expires
        or is revoked — a stale capability can never be reused. Issuance is
        audit-recorded.
        """
        grant = TimedGrant(*patterns, ttl_s=ttl_s, actor=actor, note=note,
                           clock=self._clock)
        decision = PolicyDecision(
            allowed=True,
            reason=note or f"timed grant issued ({ttl_s:g}s)",
            capability=",".join(patterns),
            actor=actor,
        )
        self._record(AUDIT_GRANT, decision,
                     {"grant_id": grant.grant_id, "ttl_s": ttl_s})
        return grant

    def revoke_grant(self, grant: "TimedGrant", *, actor: str = "",
                     note: str = "") -> None:
        """Revoke a timed grant immediately. Never raises."""
        try:
            grant.revoke()
            decision = PolicyDecision(
                allowed=True,
                reason=note or "grant revoked",
                capability=",".join(sorted(grant.patterns)),
                actor=actor,
            )
            self._record(AUDIT_REVOKE, decision, {"grant_id": grant.grant_id})
        except Exception:  # noqa: BLE001 - revocation never raises
            _log.debug("revoke_grant failed", exc_info=True)

    def child_grant(self, parent: CapabilitySet | "TimedGrant",
                    role: str) -> CapabilitySet | "TimedGrant":
        """Derive a sub-agent's grant: ``role preset ∩ parent grant``.

        Nobody grants what they do not hold — the child can never be wider
        than its parent, so privilege narrows down the agent tree by
        construction, never by call-site discipline.
        """
        return narrow_grant(parent, role)


class TimedGrant:
    """A time-bounded, revocable capability grant.

    Drop-in for :class:`CapabilitySet` as the ``grant=`` argument to
    :meth:`Policy.check`. ``grants()`` returns False once the TTL lapses or
    :meth:`revoke` is called. Thread-safe.
    """

    def __init__(self, *patterns: str, ttl_s: float, grant_id: str = "",
                 actor: str = "", note: str = "",
                 clock: Any = None) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")
        from .ids import new_short_id

        self.patterns: frozenset[str] = frozenset(patterns)
        self.ttl_s = float(ttl_s)
        self.grant_id = grant_id or new_short_id("grt_")
        self.actor = actor
        self.note = note
        self._clock = clock or _DefaultClock()
        self._issued_at = self._clock.now()
        self._revoked = False
        self._lock = threading.Lock()

    @property
    def expires_at(self) -> float:
        return self._issued_at + self.ttl_s

    @property
    def expired(self) -> bool:
        return self._clock.now() >= self.expires_at

    @property
    def revoked(self) -> bool:
        with self._lock:
            return self._revoked

    def remaining_s(self) -> float:
        return max(0.0, self.expires_at - self._clock.now())

    def revoke(self) -> None:
        with self._lock:
            self._revoked = True

    def grants(self, capability: str) -> bool:
        if self.expired or self.revoked:
            return False
        return CapabilitySet(self.patterns).grants(capability)

    def __contains__(self, capability: str) -> bool:
        return self.grants(capability)

    def __len__(self) -> int:
        return len(self.patterns)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            revoked = self._revoked
        return {
            "grant_id": self.grant_id,
            "patterns": sorted(self.patterns),
            "actor": self.actor,
            "note": self.note,
            "ttl_s": self.ttl_s,
            "expires_at": self.expires_at,
            "remaining_s": round(self.remaining_s(), 3),
            "expired": self.expired,
            "revoked": revoked,
        }


def narrow_grant(parent: CapabilitySet | TimedGrant,
                 role: str) -> CapabilitySet | TimedGrant:
    """Derive a sub-agent's grant from its parent's grant and its role.

    ``role_preset ∩ parent`` — nobody grants what they do not hold. A timed
    parent yields a timed child carrying the parent's *remaining* TTL, so a
    child can never outlive the grant that created it. Unknown roles raise
    :exc:`KeyError` via :meth:`CapabilitySet.role`.
    """
    preset = CapabilitySet.role(role)
    if isinstance(parent, TimedGrant):
        narrowed = preset.intersect(CapabilitySet(parent.patterns))
        return TimedGrant(
            *narrowed.patterns,
            ttl_s=max(1.0, parent.remaining_s()),
            actor=parent.actor,
            note=f"narrowed from {parent.grant_id} for role {role!r}",
        )
    return preset.intersect(CapabilitySet(parent.patterns))


@runtime_checkable
class PolicyStore(Protocol):
    """Persistence for permission-gradient proposals.

    The in-memory default (:class:`MemoryPolicyStore`) keeps today's
    behavior; plug a DB-backed implementation and pending proposals survive
    a process restart instead of dying with it. All methods must be
    thread-safe and never raise into the caller — the gradient treats store
    failures as "proposal not found".
    """

    def save_proposal(self, proposal: Mapping[str, Any]) -> None: ...
    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None: ...
    def pending_proposals(self) -> list[dict[str, Any]]: ...
    def update_proposal(self, proposal_id: str,
                        updates: Mapping[str, Any]) -> bool: ...


class MemoryPolicyStore:
    """In-memory :class:`PolicyStore`. The default; proposals live as long as
    the process does."""

    def __init__(self) -> None:
        self._proposals: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def save_proposal(self, proposal: Mapping[str, Any]) -> None:
        with self._lock:
            self._proposals[str(proposal["proposal_id"])] = dict(proposal)

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        with self._lock:
            proposal = self._proposals.get(proposal_id)
            return dict(proposal) if proposal else None

    def pending_proposals(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(p) for p in self._proposals.values()
                    if p.get("status") == "pending"]

    def update_proposal(self, proposal_id: str,
                        updates: Mapping[str, Any]) -> bool:
        with self._lock:
            proposal = self._proposals.get(proposal_id)
            if proposal is None:
                return False
            proposal.update(dict(updates))
            return True


def approve_with_biometric(
    policy: Policy,
    capability: str,
    *,
    title: str = "",
    timeout_s: float = 60.0,
) -> str | None:
    """Mint a confirmation token through the fingerprint prompt.

    Returns the capability-bound single-use token when the user
    authenticates, else None (biometric unavailable, prompt denied, or any
    failure — the caller falls back to the text-confirm flow). Never raises
    and never prompts implicitly: callers invoke this explicitly from an
    interactive context only.
    """
    try:
        from ..native.biometric import biometric_available, request_biometric
    except Exception:  # noqa: BLE001 - no biometric module, no biometric path
        _log.debug("biometric module unavailable for %s", capability)
        return None
    try:
        available, reason = biometric_available()
    except Exception:  # noqa: BLE001 - fail closed
        return None
    if not available:
        _log.debug("biometric unavailable (%s); caller keeps the text flow", reason)
        return None
    try:
        approved = request_biometric(title or f"approve {capability}", timeout_s=timeout_s)
    except Exception:  # noqa: BLE001 - any prompt failure is a denial
        return None
    if not approved:
        return None
    try:
        return policy.issue_confirmation(capability)
    except Exception:  # noqa: BLE001 - fail closed
        return None


class _DefaultClock:
    @staticmethod
    def now() -> float:
        return time.time()


# -- permission gradient (Dots pattern) --------------------------------------


#: Capabilities the agent may exercise autonomously without any approval:
#: read data, gather context, prepare drafts. Everything else the gradient
#: classifies as propose / act_with_approval / act_silent.
_OBSERVE_CAPABILITIES: frozenset[str] = frozenset(
    {
        Capability.FS_READ,
        Capability.MEM_READ,
        Capability.DB_READ,
        Capability.SOCIAL_READ,
        Capability.MODEL_CALL,
        Capability.NET_OUT,
        Capability.NET_BROWSER,
        Capability.NET_DOWNLOAD,
    }
)

#: Phrases that mark an explicit owner instruction. Matched case-insensitively
#: on word boundaries; a negation ("don't", "do not", ...) just before the
#: phrase, or a trailing "?", disqualifies the match.
_EXPLICIT_PATTERNS: tuple[str, ...] = (
    "post it",
    "send it",
    "do it now",
    "just do it",
    "go ahead",
    "run it",
    "submit it",
    "apply now",
    "book it",
    "buy it",
    "ship it",
    "publish it",
    "delete it",
    "execute it",
    "yes do it",
    "do it",
    "confirm",
    "confirmed",
    "approved",
    "go for it",
    "proceed",
    "make it so",
)

_NEGATION_RE = re.compile(r"\b(don't|do not|never|stop|not)\b")


def is_explicit_instruction(text: str) -> bool:
    """True when ``text`` is an explicit owner directive to execute.

    ("post it", "send it", "do it now", ...). Explicit instructions jump
    straight to execution regardless of gradient level; the gradient
    governs *autonomous* action. A trailing "?" (asking) or a negation
    ("don't post it") disqualifies the match. Never raises.
    """
    try:
        t = (text or "").strip().lower()
        if not t or t.endswith("?"):
            return False
        for pattern in _EXPLICIT_PATTERNS:
            match = re.search(r"\b" + re.escape(pattern) + r"\b", t)
            if not match:
                continue
            before = t[max(0, match.start() - 14) : match.start()]
            if _NEGATION_RE.search(before):
                continue
            return True
        return False
    except Exception:  # noqa: BLE001 - never raises
        return False


class PermissionGradient:
    """Dots-pattern permission gradient for autonomous action.

    Levels, from most to least restrained::

        observe → propose → act_with_approval → act_silent

    * **observe** — read data, gather context, prepare drafts. No approval.
    * **propose** — the agent prepares a draft and records a proposal; the
      action executes only after the owner approves the proposal (which
      mints a confirmation token through the existing flow).
    * **act_with_approval** — needs a confirmation/biometric token (existing
      :class:`Policy` flow).
    * **act_silent** — fully autonomous (explicit allow rules, owner role).

    The gradient governs *autonomous* action. An explicit owner instruction
    (see :func:`is_explicit_instruction`) bypasses the gradient entirely:
    ``check(..., explicit_override=True)`` executes per the underlying
    rules, skipping the confirmation-token gate. Deny rules and grants
    still apply, and biometric (fingerprint) approval remains mandatory.
    """

    def __init__(self, policy: Policy | None = None,
                 store: PolicyStore | None = None) -> None:
        self.policy = policy if policy is not None else Policy()
        # Proposals persist through the store: the default keeps them
        # in-memory; a DB-backed store lets pending owner approvals survive
        # a process restart.
        self.store: PolicyStore = store if store is not None else MemoryPolicyStore()
        self._lock = threading.RLock()

    def level_for(self, capability: str) -> str | None:
        """Gradient level for ``capability``.

        Returns one of the ``GRADIENT_*`` constants, or None when a deny
        rule refuses the capability outright. Conditional rules are skipped
        (no context here); :meth:`check` with context is authoritative.
        Never raises (fail closed).
        """
        try:
            with self.policy._lock:
                rules = list(self.policy._rules)
            for rule in rules:
                if not fnmatch.fnmatchcase(capability, rule.capability):
                    continue
                if rule.effect == "deny":
                    return None
                if rule.effect in ("biometric", "confirm"):
                    return GRADIENT_ACT_WITH_APPROVAL
                if rule.effect == "allow":
                    return GRADIENT_ACT_SILENT
            if capability in Capability.CONFIRMABLE or capability in Capability.BIOMETRIC:
                return GRADIENT_ACT_WITH_APPROVAL
            if capability in _OBSERVE_CAPABILITIES:
                return GRADIENT_OBSERVE
            return GRADIENT_PROPOSE
        except Exception:  # noqa: BLE001 - fail closed on evaluation errors
            return None

    def check(
        self,
        capability: str,
        *,
        actor: str = "",
        grant: CapabilitySet | None = None,
        autonomous: bool = True,
        explicit_override: bool = False,
        draft: str = "",
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """Evaluate ``capability`` under the gradient. Never raises.

        * ``explicit_override=True`` — bypass the gradient, execute per the
          underlying rules (confirmation-token gate skipped; deny rules,
          grants, and biometric approval still enforced).
        * ``autonomous=True`` (default) — the propose level records a
          proposal and returns a denied-pending-approval decision carrying
          its id.
        * ``autonomous=False`` — fall back to a plain policy check (the
          caller is handling a user request, not acting on its own).
        """
        try:
            if explicit_override:
                decision = self.policy.check(
                    capability,
                    actor=actor,
                    grant=grant,
                    context=context,
                    explicit_override=True,
                )
                decision.gradient = GRADIENT_ACT_SILENT
                decision.explicit_override = True
                return decision

            level = self.level_for(capability)
            if level is None:
                decision = self.policy.check(
                    capability, actor=actor, grant=grant, context=context
                )
                decision.gradient = ""
                return decision

            if level == GRADIENT_PROPOSE and autonomous:
                proposal = self.propose(
                    capability,
                    actor=actor,
                    draft=draft or f"autonomous {capability}",
                    context=context,
                )
                decision = PolicyDecision(
                    allowed=False,
                    reason=(
                        "autonomous action requires owner approval "
                        f"(proposal {proposal['proposal_id']})"
                    ),
                    capability=capability,
                    actor=actor,
                    needs_proposal=True,
                    proposal_id=proposal["proposal_id"],
                    gradient=GRADIENT_PROPOSE,
                )
                self.policy._record(AUDIT_PROPOSE, decision, context)
                return decision

            decision = self.policy.check(
                capability,
                actor=actor,
                grant=grant,
                context=context,
                audit_kind=AUDIT_OBSERVE if level == GRADIENT_OBSERVE else None,
            )
            decision.gradient = level
            return decision
        except Exception as exc:  # noqa: BLE001 - never raises, fail closed
            return PolicyDecision(
                allowed=False,
                reason=f"gradient evaluation failed: {exc}",
                capability=str(capability),
                actor=actor,
            )

    # -- proposals -----------------------------------------------------------
    def propose(
        self,
        capability: str,
        *,
        actor: str = "",
        draft: str = "",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a proposal for an autonomous action. Returns the proposal."""
        from .ids import new_short_id

        proposal_id = new_short_id("prp_")
        proposal = {
            "proposal_id": proposal_id,
            "capability": capability,
            "actor": actor,
            "draft": draft,
            "status": "pending",
            "ts": time.time(),
            "context": dict(context or {}),
        }
        try:
            self.store.save_proposal(proposal)
        except Exception:  # noqa: BLE001 - never raises
            pass
        return dict(proposal)

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        try:
            return self.store.get_proposal(proposal_id)
        except Exception:  # noqa: BLE001 - never raises
            return None

    def pending_proposals(self) -> list[dict[str, Any]]:
        try:
            return self.store.pending_proposals()
        except Exception:  # noqa: BLE001 - never raises
            return []

    def approve_proposal(self, proposal_id: str) -> str | None:
        """Owner approves a proposal → mint the confirmation token.

        The agent then passes the token to :meth:`Policy.check` and the
        action executes through the existing confirmation flow. Returns
        None for unknown/already-resolved proposals. Never raises.
        """
        try:
            proposal = self.store.get_proposal(proposal_id)
            if proposal is None or proposal.get("status") != "pending":
                return None
            capability = proposal.get("capability", "")
            if not capability:
                return None
            token = self.policy.issue_confirmation(capability)
            self.store.update_proposal(proposal_id, {"status": "approved",
                                                     "token_issued": True})
            return token
        except Exception:  # noqa: BLE001 - never raises
            return None

    def reject_proposal(self, proposal_id: str, *, note: str = "") -> bool:
        """Owner rejects a proposal. Never raises."""
        try:
            proposal = self.store.get_proposal(proposal_id)
            if proposal is None or proposal.get("status") != "pending":
                return False
            updates: dict[str, Any] = {"status": "rejected"}
            if note:
                updates["reject_note"] = note
            return bool(self.store.update_proposal(proposal_id, updates))
        except Exception:  # noqa: BLE001 - never raises
            return False
