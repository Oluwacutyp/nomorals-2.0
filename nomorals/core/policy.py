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
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "AUDIT_BIOMETRIC",
    "AUDIT_DENY",
    "Capability",
    "CapabilitySet",
    "Policy",
    "PolicyDecision",
    "approve_with_biometric",
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


@dataclass
class PolicyDecision:
    """Result of a policy check."""

    allowed: bool
    reason: str = ""
    capability: str = ""
    actor: str = ""
    needs_confirmation: bool = False
    needs_biometric: bool = False
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
            "audit_id": self.audit_id,
        }


@dataclass
class _Rule:
    capability: str
    effect: str  # allow | deny | confirm | biometric
    note: str = ""
    priority: int = 0


class Policy:
    """Evaluates capability requests against rules, grants, and confirmations.

    Evaluation order:
      1. explicit ``deny`` rules
      2. ``biometric`` / ``confirm`` rules → require a valid confirmation token
         (biometric rules additionally require the token to be minted through
         the fingerprint prompt)
      3. explicit ``allow`` rules
      4. the actor's :class:`CapabilitySet`
      5. default deny
    """

    def __init__(
        self,
        *,
        default_grant: CapabilitySet | None = None,
        enforce: bool = True,
        confirmation_ttl: float = 300.0,
        clock: Any = None,
    ) -> None:
        self.default_grant = default_grant if default_grant is not None else CapabilitySet.none()
        self.enforce = enforce
        self.confirmation_ttl = confirmation_ttl
        self._rules: list[_Rule] = []
        # token -> (expiry_ts, capability it was minted for)
        self._confirmations: dict[str, tuple[float, str]] = {}
        self._audit: list[dict[str, Any]] = []
        self._audit_limit = 2000
        self._lock = threading.RLock()
        self._clock = clock or _DefaultClock()
        self._counts = {"allow": 0, "deny": 0, "confirm": 0, "biometric": 0}

    # -- rule management -----------------------------------------------------
    def allow(self, capability: str, *, note: str = "", priority: int = 10) -> Policy:
        return self._add(_Rule(capability, "allow", note, priority))

    def deny(self, capability: str, *, note: str = "", priority: int = 100) -> Policy:
        return self._add(_Rule(capability, "deny", note, priority))

    def confirm(self, capability: str, *, note: str = "", priority: int = 50) -> Policy:
        return self._add(_Rule(capability, "confirm", note, priority))

    def biometric(self, capability: str, *, note: str = "", priority: int = 60) -> Policy:
        """Gate on fingerprint approval: the confirmation token for this
        capability must be minted through :func:`approve_with_biometric`."""
        return self._add(_Rule(capability, "biometric", note, priority))

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
        the :attr:`Capability.BIOMETRIC` set applies. Never raises.
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

    # -- evaluation ----------------------------------------------------------
    def check(
        self,
        capability: str,
        *,
        actor: str = "",
        grant: CapabilitySet | None = None,
        confirmation: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        """Evaluate whether ``actor`` may exercise ``capability``."""
        effective = grant if grant is not None else self.default_grant
        reason = ""
        needs_confirm = False
        needs_biometric = False

        with self._lock:
            for rule in self._rules:
                if not fnmatch.fnmatchcase(capability, rule.capability):
                    continue
                if rule.effect == "deny":
                    decision = PolicyDecision(
                        allowed=False,
                        reason=rule.note or f"denied by policy rule: {rule.capability}",
                        capability=capability,
                        actor=actor,
                    )
                    self._record(AUDIT_DENY, decision, context)
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
        confirmable = needs_confirm or capability in Capability.CONFIRMABLE
        biometric_required = needs_biometric or capability in Capability.BIOMETRIC

        if not self.enforce:
            decision = PolicyDecision(
                allowed=True,
                reason="policy enforcement disabled",
                capability=capability,
                actor=actor,
            )
            self._record(AUDIT_ALLOW, decision, context)
            return decision

        if not granted:
            decision = PolicyDecision(
                allowed=False,
                reason=f"actor {actor or 'anonymous'!r} lacks capability {capability!r}",
                capability=capability,
                actor=actor,
            )
            self._record(AUDIT_DENY, decision, context)
            return decision

        if confirmable and (
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
            )
            self._record(
                AUDIT_BIOMETRIC if biometric_required else AUDIT_CONFIRM,
                decision,
                context,
            )
            return decision

        decision = PolicyDecision(
            allowed=True,
            reason="granted",
            capability=capability,
            actor=actor,
        )
        self._record(AUDIT_ALLOW, decision, context)
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
