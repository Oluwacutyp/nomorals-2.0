"""Police — the seven-rung enforcement ladder.

CONSTITUTION (test-enforced):

* The immune system may only REVOKE capability, never ACQUIRE it.
  There is deliberately NO grant / acquire method on :class:`Police`.
* Never reduce the OWNER's capability. Owner-directed work gets a
  REPORT, never a block — the same posture as ``finance/guard.py``
  (warning, never blocking).
* Enforcement teeth apply ONLY to the system's own runaway processes,
  never to owner-directed work and never to owner work.
* Every action leaves a receipt (:class:`Receipt`).

Rungs: 1 Observe · 2 Mark · 3 Quarantine · 4 Disarm · 5 Arrest ·
6 Case file · 7 Notify.

All side effects go through INJECTED callables (``revoke_fn``,
``vault_fn``, ``kill_fn``, ``cancel_fn``, ``presence_fn``,
``digest_fn``, ``casefile_fn``, ``enforce_fn``). Police never imports
``core/policy.py``, ``os/replay.py`` or anything else: the caller
wires the real functions in. A rung with no injected function still
leaves a receipt, but performs no side effect.

EFFECT HONESTY (doc 20 P2): a receipt may claim only an effect that
a wired function actually performed during that action, with the
function's outcome carried on the record. Every effect rung records
``details["effect"]`` (``performed`` / ``partial`` /
``not_performed`` / ``failed``) and ``details["effect_outcomes"]``
(one entry per function: whether it was wired, and when it ran, its
return value or the exception it raised). Where nothing was wired
or the function failed, the reason text states the true outcome in
plain words — never the success wording.

Also in this module: :func:`consult_then_enforce` (consult-then-enforce
resources gate) and :func:`redteam_selftest`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "RUNGS",
    "RUNG_NAMES",
    "CANCER_MAX_PROCESSES",
    "Receipt",
    "mark_tainted",
    "Police",
    "consult_then_enforce",
    "redteam_selftest",
]

RUNGS: Dict[int, str] = {
    1: "Observe",
    2: "Mark",
    3: "Quarantine",
    4: "Disarm",
    5: "Arrest",
    6: "Case file",
    7: "Notify",
}
RUNG_NAMES = dict(RUNGS)

#: Cancer threshold: a system process spawning at/above this many
#: processes is treated as unbounded growth (see consult_then_enforce).
CANCER_MAX_PROCESSES = 100


@dataclass
class Receipt:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    rung: int = 1
    action: str = "report_only"
    target: str = ""
    reason: str = ""
    ts: float = field(default_factory=time.time)
    actor: str = "immune.police"
    reversible: bool = True
    details: Dict[str, Any] = field(default_factory=dict)


def mark_tainted(text: str) -> str:
    """Wrap untrusted content in tainted delimiters (rung 2)."""
    return f"[UNTRUSTED-DATA]{text}[/UNTRUSTED-DATA]"


def _event_field(event: Any, *names: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        for name in names:
            if event.get(name) is not None:
                return event.get(name)
        return default
    for name in names:
        if getattr(event, name, None) is not None:
            return getattr(event, name)
    return default


def _call_maybe_with_event(fn: Callable, event: Any) -> Any:
    """Call an injected fn with the event, falling back to no-arg."""
    try:
        return fn(event)
    except TypeError:
        return fn()


def _recordable(value: Any) -> Any:
    """A function result as a receipt may carry it: JSON scalars
    verbatim, anything else as a bounded repr."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)[:200]


def _run_effect(fn_name: str, fn: Callable | None, *args: Any) -> dict[str, Any]:
    """Run one injected effect function and record what happened.

    The returned entry is the receipt's evidence for the effect: it
    names the function, whether it was wired at all, and — when it
    ran — its outcome (return value, or the exception it raised).
    A raised exception is captured, never propagated: the failure
    IS the outcome the receipt must record.
    """
    if fn is None:
        return {"fn": fn_name, "wired": False}
    try:
        result = fn(*args)
    except Exception as exc:  # noqa: BLE001 - the failure is the outcome
        return {"fn": fn_name, "wired": True, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}
    return {"fn": fn_name, "wired": True, "ok": True,
            "result": _recordable(result)}


def _effect_of(outcomes: list[dict[str, Any]]) -> str:
    """Fold per-function outcomes into one effect word."""
    ran_ok = [o for o in outcomes if o.get("wired") and o.get("ok")]
    ran_bad = [o for o in outcomes if o.get("wired") and not o.get("ok")]
    if not ran_ok and not ran_bad:
        return "not_performed"
    if ran_ok and not ran_bad and len(ran_ok) == len(outcomes):
        return "performed"
    if ran_ok:
        return "partial"
    return "failed"


class Police:
    """Revoke-only enforcement. No grant/acquire method exists — by design."""

    def __init__(
        self,
        *,
        revoke_fn: Callable[[str], Any] | None = None,
        vault_fn: Callable[[Any], Any] | None = None,
        kill_fn: Callable[[str], Any] | None = None,
        cancel_fn: Callable[[str], Any] | None = None,
        presence_fn: Callable[..., Any] | None = None,
        digest_fn: Callable[[Any], Any] | None = None,
        casefile_fn: Callable[..., Any] | None = None,
        enforce_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        self.revoke_fn = revoke_fn
        self.vault_fn = vault_fn
        self.kill_fn = kill_fn
        self.cancel_fn = cancel_fn
        self.presence_fn = presence_fn
        self.digest_fn = digest_fn
        self.casefile_fn = casefile_fn
        self.enforce_fn = enforce_fn

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _receipt(
        rung: int, action: str, target: str, reason: str, *, reversible: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> Receipt:
        return Receipt(
            rung=rung, action=action, target=str(target or ""),
            reason=reason, reversible=reversible, details=dict(details or {}),
        )

    def _resolve(self, per_call: Any, stored: Any) -> Any:
        return per_call if per_call is not None else stored

    # ── rungs 1–2 ──────────────────────────────────────────────────

    def observe(self, event: Any, reason: str = "") -> Receipt:
        return self._receipt(
            1, "observe", _event_field(event, "target", "id", default=""),
            reason or "observed, no effect",
            details={"event": event if isinstance(event, dict) else str(event)},
        )

    def mark_text(self, text: str) -> str:
        return mark_tainted(text)

    # ── rung 3: quarantine — vault, NEVER delete ────────────────────

    def quarantine(
        self, artifact: Any, vault_fn: Callable[[Any], Any] | None = None
    ) -> Receipt:
        fn = self._resolve(vault_fn, self.vault_fn)
        outcome = _run_effect("vault_fn", fn, artifact)
        effect = _effect_of([outcome])
        # Mark the original as quarantined only when the vault
        # actually took it — it still exists; nothing is deleted.
        if effect == "performed" and isinstance(artifact, dict):
            artifact["quarantined"] = True
        target = _event_field(artifact, "id", default="") if isinstance(artifact, dict) else str(artifact)[:80]
        if effect == "performed":
            reason = "artifact moved to vault, never deleted"
        elif effect == "failed":
            reason = (f"vault function failed ({outcome['error']}) — "
                      "artifact left in place, unmarked")
        else:
            reason = ("artifact not moved to vault — no vault function "
                      "wired; artifact left in place, unmarked")
        return self._receipt(
            3, "quarantine", target, reason,
            reversible=True,
            details={"artifact_id": target, "effect": effect,
                     "effect_outcomes": [outcome]},
        )

    # ── rung 4: disarm — revoke a capability grant ──────────────────

    def disarm(
        self, grant_id: str, revoke_fn: Callable[[str], Any] | None = None
    ) -> Receipt:
        fn = self._resolve(revoke_fn, self.revoke_fn)
        outcome = _run_effect("revoke_fn", fn, grant_id)
        effect = _effect_of([outcome])
        if effect == "performed":
            reason = f"capability grant {grant_id} revoked"
        elif effect == "failed":
            reason = (f"revocation of grant {grant_id} failed — revoke "
                      f"function raised {outcome['error']}; "
                      "grant left in force")
        else:
            reason = (f"no revocation performed — revoke function not "
                      f"wired; grant {grant_id} left in force")
        return self._receipt(
            4, "disarm", grant_id, reason,
            reversible=True,
            details={"grant_id": grant_id, "effect": effect,
                     "effect_outcomes": [outcome]},
        )

    # ── rung 5: arrest — kill process group + cancel mission ────────

    def arrest(
        self,
        process_id: str,
        kill_fn: Callable[[str], Any] | None = None,
        cancel_fn: Callable[[str], Any] | None = None,
    ) -> Receipt:
        kill = self._resolve(kill_fn, self.kill_fn)
        cancel = self._resolve(cancel_fn, self.cancel_fn)
        outcomes = [
            _run_effect("kill_fn", kill, process_id),
            _run_effect("cancel_fn", cancel, process_id),
        ]
        effect = _effect_of(outcomes)
        kill_out, cancel_out = outcomes
        if kill_out.get("ok"):
            kill_part = f"process {process_id} killed"
        elif kill_out.get("wired"):
            kill_part = (f"kill of process {process_id} failed "
                         f"({kill_out['error']})")
        else:
            kill_part = (f"process {process_id} not killed — "
                         "no kill function wired")
        if cancel_out.get("ok"):
            cancel_part = "mission cancelled"
        elif cancel_out.get("wired"):
            cancel_part = f"mission cancel failed ({cancel_out['error']})"
        else:
            cancel_part = ("mission not cancelled — "
                           "no cancel function wired")
        return self._receipt(
            5, "arrest", process_id,
            f"{kill_part}; {cancel_part}",
            reversible=False,
            details={"process_id": process_id, "effect": effect,
                     "effect_outcomes": outcomes},
        )

    # ── rung 6: case file ──────────────────────────────────────────

    def build_case(
        self,
        event: Any,
        casefile_fn: Callable[..., Any] | None = None,
        timeline_fn: Callable[..., Any] | None = None,
    ) -> Receipt:
        fn = self._resolve(casefile_fn, self.casefile_fn)
        case_id: Any = None
        if fn is None:
            outcome: dict[str, Any] = {"fn": "casefile_fn", "wired": False}
        else:
            try:
                try:
                    result = fn(event, timeline_fn)
                except TypeError:
                    result = fn(event)
                case_id = getattr(result, "id", None)
                if case_id is None and isinstance(result, dict):
                    case_id = result.get("id")
                outcome = {"fn": "casefile_fn", "wired": True, "ok": True,
                           "result": _recordable(case_id)}
            except Exception as exc:  # noqa: BLE001 - the failure is the outcome
                case_id = None
                outcome = {"fn": "casefile_fn", "wired": True, "ok": False,
                           "error": f"{type(exc).__name__}: {exc}"}
        effect = _effect_of([outcome])
        if effect == "performed" and case_id:
            reason = "case file built"
        elif effect == "performed":
            reason = ("case file function ran but produced no case id — "
                      "nothing citable was filed")
        elif effect == "failed":
            reason = (f"case file build failed — casefile function "
                      f"raised {outcome['error']}")
        else:
            reason = "no case file built — no casefile function wired"
        return self._receipt(
            6, "casefile", _event_field(event, "target", "id", default=""),
            reason,
            reversible=True,
            details={"case_id": case_id, "effect": effect,
                     "effect_outcomes": [outcome]},
        )

    # ── rung 7: notify — interrupt only if interruptible ────────────

    def notify(
        self,
        event: Any,
        presence_fn: Callable[..., Any] | None = None,
        digest_fn: Callable[[Any], Any] | None = None,
        *,
        rung: int = 7,
    ) -> Receipt:
        presence_source = self._resolve(presence_fn, self.presence_fn)
        digest = self._resolve(digest_fn, self.digest_fn)
        interruptible = True
        if presence_source is not None:
            presence = _call_maybe_with_event(presence_source, event)
            if isinstance(presence, dict):
                interruptible = bool(presence.get("interruptible", True))
            else:
                interruptible = bool(presence)
        target = _event_field(event, "target", "id", default="")
        if rung >= 4 and not interruptible:
            outcome = _run_effect("digest_fn", digest, event)
            effect = _effect_of([outcome])
            if effect == "performed":
                return self._receipt(
                    rung, "notify_batched", target,
                    "not interruptible — batched to digest, no ping "
                    "storm (digest function called; outcome in details)",
                    details={"delivered": "digest", "effect": effect,
                             "effect_outcomes": [outcome]},
                )
            if effect == "failed":
                return self._receipt(
                    rung, "notify_batched", target,
                    f"not interruptible — digest function failed "
                    f"({outcome['error']}); nothing delivered",
                    details={"delivered": "none", "effect": effect,
                             "effect_outcomes": [outcome]},
                )
            return self._receipt(
                rung, "notify_batched", target,
                "not interruptible — digest batching requested but no "
                "digest function wired; nothing delivered",
                details={"delivered": "none", "effect": effect,
                         "effect_outcomes": [outcome]},
            )
        # Police performs no delivery itself — no delivery function
        # is wired to it. This receipt is the DECISION record
        # (interrupt now); the caller delivers.
        return self._receipt(
            rung, "notify", target,
            "notification decided immediate (interruptible) — Police "
            "performs no delivery itself (no delivery function is "
            "wired to it); this receipt is the decision record and "
            "the caller delivers",
            details={"delivered": "decision_immediate",
                     "effect": "not_performed",
                     "effect_outcomes": [{"fn": "delivery",
                                          "wired": False}]},
        )

    # ── dispatch ───────────────────────────────────────────────────

    def respond(
        self,
        event: Any,
        *,
        owner_directed: bool = False,
        target_kind: str | None = None,
        rung: int | None = None,
        reason: str = "",
        **overrides: Any,
    ) -> Receipt:
        """Respond to an event under the constitution.

        Owner-directed work — or work whose target is owner work —
        returns a REPORT-ONLY receipt and takes NO rung 3–5 action,
        never blocks. System processes may escalate.
        """
        event_owner = bool(_event_field(event, "owner_directed", default=False))
        event_target_kind = _event_field(event, "target_kind", default=None)
        effective_target = target_kind or event_target_kind
        is_owner_work = (
            owner_directed
            or event_owner
            or (isinstance(effective_target, str) and effective_target in ("owner", "owner_work", "owner-directed"))
        )
        target = _event_field(
            event, "target", "process_id", "grant_id", "id", default=""
        )
        if is_owner_work:
            return self._receipt(
                1, "report_only", target,
                reason or "owner-directed / owner work: report never block",
                reversible=True,
                details={"blocked": False, "owner_directed": True},
            )
        chosen = rung if rung is not None else _event_field(event, "rung", "requested_rung", default=1)
        try:
            chosen = int(chosen)
        except (TypeError, ValueError):
            chosen = 1
        if chosen not in RUNGS:
            raise ValueError(f"unknown police rung {chosen}; rungs are {sorted(RUNGS)}")
        if chosen == 1:
            return self.observe(event, reason or "observed, no effect")
        if chosen == 2:
            text = _event_field(event, "content", "text", default="") or ""
            marked = mark_tainted(str(text))
            return self._receipt(
                2, "mark", target, reason or "content marked untrusted",
                details={"marked_text": marked},
            )
        if chosen == 3:
            artifact = _event_field(event, "artifact", default=event)
            return self.quarantine(artifact, overrides.get("vault_fn"))
        if chosen == 4:
            grant_id = _event_field(event, "grant_id", "target", default="") or str(target)
            return self.disarm(str(grant_id), overrides.get("revoke_fn"))
        if chosen == 5:
            process_id = _event_field(event, "process_id", "target", default="") or str(target)
            return self.arrest(
                str(process_id), overrides.get("kill_fn"), overrides.get("cancel_fn")
            )
        if chosen == 6:
            return self.build_case(
                event, overrides.get("casefile_fn"), overrides.get("timeline_fn")
            )
        # rung 7
        return self.notify(
            event, overrides.get("presence_fn"), overrides.get("digest_fn"), rung=7
        )

    # ── consult-then-enforce (resources gate) ──────────────────────

    def consult_then_enforce(
        self,
        sample: Dict[str, Any],
        *,
        is_system_process: bool = False,
        owner_directed: bool = False,
        enforce_fn: Callable[[Any], Any] | None = None,
    ) -> Receipt:
        return consult_then_enforce(
            sample,
            is_system_process=is_system_process,
            owner_directed=owner_directed,
            enforce_fn=enforce_fn if enforce_fn is not None else self.enforce_fn,
        )


def _exceeds_cancer(sample: Dict[str, Any]) -> bool:
    spawned = sample.get("processes_spawned", 0) or 0
    growth = sample.get("process_growth", sample.get("growth", 0)) or 0
    if sample.get("unbounded") is True:
        return True
    if spawned >= CANCER_MAX_PROCESSES:
        return True
    if spawned >= 20 and growth >= 3:
        return True
    return False


def consult_then_enforce(
    sample: Dict[str, Any],
    *,
    is_system_process: bool = False,
    owner_directed: bool = False,
    enforce_fn: Callable[[Any], Any] | None = None,
) -> Receipt:
    """Consult-then-enforce resources gate.

    Owner-directed work: report only, NEVER enforce. A system process
    over the cancer thresholds (unbounded process spawning / growth):
    call the injected ``enforce_fn`` and leave a receipt. Anything
    else: report only. This function never touches owner work.
    """
    if owner_directed:
        return Receipt(
            rung=1, action="report_only", target=str(sample.get("process_id", "")),
            reason="owner-directed: report only, never enforce",
            details={"blocked": False, "enforced": False},
        )
    if is_system_process and _exceeds_cancer(sample or {}):
        outcome = _run_effect("enforce_fn", enforce_fn, sample)
        effect = _effect_of([outcome])
        details: dict[str, Any] = {
            "enforced": effect == "performed",
            "effect": effect,
            "effect_outcomes": [outcome],
            "sample": dict(sample or {}),
        }
        if effect == "performed":
            reason = "system process over cancer thresholds: enforced"
        elif effect == "failed":
            reason = ("system process over cancer thresholds: enforce "
                      f"function failed ({outcome['error']}) — "
                      "no effect taken")
        else:
            reason = ("system process over cancer thresholds: "
                      "enforcement requested but no enforce function "
                      "wired — no effect taken")
        return Receipt(
            rung=5, action="enforce", target=str(sample.get("process_id", "")),
            reason=reason,
            reversible=False, details=details,
        )
    return Receipt(
        rung=1, action="report_only", target=str(sample.get("process_id", "")),
        reason="below enforcement thresholds: report only",
        details={"blocked": False, "enforced": False},
    )


def redteam_selftest(police: "Police") -> Dict[str, Any]:
    """Self-test the police against the immune-evasion class.

    Checks: no grant/acquire method exists, and an owner-directed
    event returns report-only (no rung 3–5 action). Returns
    ``{"passed": bool, "checks": [...]}`` — data, not a verdict engine.
    """
    checks: List[Dict[str, Any]] = []

    def _check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for forbidden in ("grant", "acquire", "grant_capability", "acquire_capability"):
        _check(
            f"no {forbidden} method",
            not hasattr(police, forbidden),
            f"Police must never expose {forbidden} (revoke-only)",
        )
    # Owner-directed must be report-only even at a high rung.
    calls: List[str] = []
    probe = Police(
        revoke_fn=lambda gid: calls.append(f"revoke:{gid}"),
        kill_fn=lambda pid: calls.append(f"kill:{pid}"),
        cancel_fn=lambda pid: calls.append(f"cancel:{pid}"),
        vault_fn=lambda artifact: calls.append("vault"),
    )
    receipt = probe.respond(
        {"target": "owner-task", "grant_id": "g1", "process_id": "p1"},
        owner_directed=True, rung=5,
    )
    _check(
        "owner-directed returns report-only",
        receipt.action == "report_only" and receipt.rung == 1,
        f"got action={receipt.action} rung={receipt.rung}",
    )
    _check(
        "owner-directed triggers no enforcement side effect",
        calls == [],
        f"side effects fired: {calls}",
    )
    # Consult-then-enforce must also refuse owner-directed enforcement.
    enforced: List[Any] = []
    receipt2 = consult_then_enforce(
        {"processes_spawned": 10_000, "process_id": "p9"},
        is_system_process=True, owner_directed=True,
        enforce_fn=lambda s: enforced.append(s),
    )
    _check(
        "consult_then_enforce never enforces owner-directed work",
        receipt2.action == "report_only" and enforced == [],
        f"got action={receipt2.action}, enforced={enforced}",
    )
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
