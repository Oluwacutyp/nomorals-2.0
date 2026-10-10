"""Power mode: the owner's documented, audited expansion of capability.

This is the "Ultimate" tier, done the honest way: nothing hidden, nothing
cruel, everything reversible. The owner unlocks power mode either with the key
they set in ``partner.owner_key`` (env ``NM_PARTNER_OWNER_KEY``) or with the
owner seal — one of their ingrained identities plus the passphrase baked via
``nm owner seal`` (``nomorals.core.owner``). Either way the runtime wakes her
up wider — more autonomy, more parallel chats, longer context budgets, bolder
group posting, shorter typing delays.

Deliberate invariants:

* The key is **never printed** in status, logs, or replies. Comparisons use
  :func:`hmac.compare_digest`.
* Default posture (``partner.power_default_on``, env
  ``NM_PARTNER_POWER_DEFAULT_ON``, default true): power mode **auto-activates**
  on first use — the owner's standing choice is unlimited/unrestricted
  operation.  No key required for the default; the auto-activation is
  audited like any unlock.
* An explicit ``lock`` persists a locked state that survives reboots until
  the owner unlocks again — the owner's explicit choice always wins over
  the default.
* With ``power_default_on=false`` and no key configured, power mode can
  never activate, no matter what is typed.
* Every unlock/lock/auto-activation is written to ``audit_log`` with the actor.
* Unlocking changes *dials*, not *constraints*: rate limits, the owner-chat
  trust rules, and the persona's voice are untouched.
* Locking restores the configured values exactly.

Power ON means unlimited/unrestricted: no volume caps, autonomy cascaded on,
the improvement auto-tick on.  Power OFF (explicitly locked) is the limited
profile: the configured protective defaults stay in force — bounded
parallelism, bounded proactive sends, suggest-mode autonomy.  See
:meth:`PowerMode.limited_profile`.
"""

from __future__ import annotations

import hmac
import json
import time
from dataclasses import dataclass, replace
from typing import Any

from ..core.ids import ulid_now
from ..core.logging_setup import get_logger
from ..core.owner import seal_configured, verify_owner
from ..storage.kv import KVStore


# Shim classes for settings that were removed from core.config
# These provide minimal defaults for power mode operations
@dataclass(frozen=True)
class AutonomySettings:
    """Minimal autonomy settings for power mode."""
    enabled: bool = False


@dataclass(frozen=True)
class ImprovementSettings:
    """Minimal improvement settings for power mode."""
    auto_tick: bool = False

_log = get_logger(__name__)

AUDIT_ACTOR = "power-mode"
ACTIVE_KEY = "partner.power"


@dataclass(frozen=True)
class PowerChanges:
    """What power mode changes, so status can report it honestly.

    ``section`` names the settings node the field lives on: ``partner``
    (the companion dials) or ``root`` (system-wide dials like the
    autonomous cascade and the multi-model router).
    """

    field: str
    before: Any
    after: Any
    section: str = "partner"

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "before": self.before,
                "after": self.after, "section": self.section}


class PowerMode:
    """Holds power-mode state and applies/releases the dial changes."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self._active = False
        self._restored: list[PowerChanges] = []
        self.unlocked_by: str = ""
        self.unlocked_at: float = 0.0

    # ── state ────────────────────────────────────────────────────────────────
    @property
    def active(self) -> bool:
        return self._active

    def _audit(self, action: str, detail: dict[str, Any]) -> None:
        db = getattr(self.context, "db", None)
        if db is None:
            return
        try:
            db.execute(
                "INSERT INTO audit_log (id, ts, actor, action, capability, decision, detail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ulid_now(),
                    time.time(),
                    AUDIT_ACTOR,
                    action,
                    "power-mode",
                    "allow" if action == "power.unlock" else "info",
                    json.dumps(detail),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - audit must never break a reply
            _log.warning("power-mode audit failed: %s", exc)

    # ── unlock / lock ────────────────────────────────────────────────────────
    def unlock(self, key: str, *, actor: str = "console",
               identity: str = "") -> dict[str, Any]:
        """Try to unlock with ``key``. Returns a report dict; the key itself
        never appears in it.

        Two ways in, tried in order:

        1. the env/config owner key (``partner.owner_key``) — existing path;
        2. the owner seal — ``identity`` names one of the ingrained owner
           identities and ``key`` is the passphrase baked via ``nm owner
           seal`` (``nomorals.core.owner``).
        """
        expected = str(getattr(self.context.settings.partner, "owner_key", "") or "")
        if expected and hmac.compare_digest(key, expected):
            return self._activate(actor, via="owner-key")
        if identity and seal_configured() and verify_owner(identity, key):
            return self._activate(actor, via="owner-seal")
        if not expected and not seal_configured():
            self._audit("power.unlock_denied", {"actor": actor, "why": "no unlock configured"})
            return {
                "ok": False,
                "message": (
                    "no unlock is configured — set partner.owner_key "
                    "(NM_PARTNER_OWNER_KEY) or bake an owner seal with "
                    "`nm owner seal`. Either stays on your machine."
                ),
            }
        self._audit("power.unlock_denied", {"actor": actor, "why": "bad key"})
        return {"ok": False, "message": "wrong key. power mode stays locked."}

    def _activate(self, actor: str, *, via: str) -> dict[str, Any]:
        """Widen the dials after a successful unlock (either path)."""
        if self._active:
            return {"ok": True, "message": "power mode is already active.", "changes": []}
        changes = self._widen()
        self._restored = changes
        self._active = True
        self.unlocked_by = actor
        self.unlocked_at = time.time()
        self._persist(True, actor)
        self._audit(
            "power.unlock",
            {"actor": actor, "via": via,
             "fields": [c.field for c in changes]},
        )
        _log.info("power mode UNLOCKED by %s via %s: %s",
                  actor, via, [c.field for c in changes])
        return {
            "ok": True,
            "message": f"power mode active. widened: {', '.join(c.field for c in changes)}.",
            "changes": [c.to_dict() for c in changes],
        }

    def lock(self, *, actor: str = "console") -> dict[str, Any]:
        was_active = self._active
        if was_active:
            self._restore()
            self._active = False
        # Persist an explicit lock: it survives reboots and beats the
        # default-on posture until the owner unlocks again.
        self._persist(False, actor, locked=True)
        self._audit("power.lock", {"actor": actor})
        _log.info("power mode LOCKED by %s", actor)
        return {
            "ok": True,
            "message": (
                "power mode locked; configured values restored."
                if was_active
                else "power mode locked."
            ),
        }

    def adopt_persisted_state(self) -> bool:
        """Re-apply a power mode that was unlocked in an earlier process.

        The key was already verified at unlock time; restoring on a new
        machine boot is the owner's own machine re-honoring the owner's
        choice. Widens the dials again from the base config and journals it.
        An explicit lock is honored — it is never overridden here.
        """
        if self._active:
            return False
        persisted = self._read_persisted()
        if not persisted or not persisted.get("active"):
            return False
        changes = self._widen()
        self._restored = changes
        self._active = True
        self.unlocked_by = str(persisted.get("by") or "owner")
        self.unlocked_at = float(persisted.get("ts", 0.0))
        self._audit("power.restore", {"actor": self.unlocked_by,
                                      "fields": [c.field for c in changes]})
        _log.info("power mode restored from persisted state (by %s)", self.unlocked_by)
        return True

    def limited_profile(self) -> dict[str, Any]:
        """The concrete limits in force while power mode is OFF.

        This is the "a bit limited" profile: the configured protective
        defaults, reported honestly so status can show what is capped.
        """
        partner = self.context.settings.partner
        return {
            "autonomy_mode": getattr(partner, "autonomy_mode", "suggest"),
            "max_parallel_chats": getattr(partner, "max_parallel_chats", 5),
            "max_proactive_dm_per_day": getattr(partner, "max_proactive_dm_per_day", 10),
            "max_group_posts_per_day": getattr(partner, "max_group_posts_per_day", 5),
            "history_window": getattr(partner, "history_window", 50),
        }

    def _persist(self, active: bool, actor: str, *, locked: bool = False) -> None:
        db = getattr(self.context, "db", None)
        if db is None:
            return
        try:
            KVStore(db).set(ACTIVE_KEY, {"active": bool(active), "locked": bool(locked),
                                         "by": actor, "ts": time.time()})
        except Exception as exc:  # noqa: BLE001
            _log.warning("power-mode persist failed: %s", exc)

    def _read_persisted(self) -> dict[str, Any] | None:
        db = getattr(self.context, "db", None)
        if db is None:
            return None
        try:
            data = KVStore(db).get(ACTIVE_KEY)
            if not isinstance(data, dict):
                return None
            return data
        except Exception:  # noqa: BLE001
            return None

    def _persisted_locked(self) -> bool:
        data = self._read_persisted()
        return bool(data and data.get("locked") and not data.get("active"))

    def _default_on(self) -> bool:
        return bool(getattr(self.context.settings.partner, "power_default_on", True))

    def _auto_activate(self) -> bool:
        """Activate from the default-on posture.  Returns True when activated."""
        if self._active:
            return False
        if not self._default_on():
            return False
        if self._persisted_locked():
            return False
        changes = self._widen()
        self._restored = changes
        self._active = True
        self.unlocked_by = "default"
        self.unlocked_at = time.time()
        self._persist(True, "default")
        self._audit("power.auto_activate",
                    {"actor": "default",
                     "fields": [c.field for c in changes]})
        _log.info("power mode AUTO-ACTIVATED (default-on): %s",
                  [c.field for c in changes])
        return True

    def status(self) -> dict[str, Any]:
        if self._active:
            return {
                "active": True,
                "applied_in_process": True,
                "unlocked_by": self.unlocked_by,
                "unlocked_at": self.unlocked_at,
                "default_on": self._default_on(),
                "key_configured": bool(str(getattr(self.context.settings.partner, "owner_key", "") or ""))
                or seal_configured(),
                "changes": [c.to_dict() for c in self._restored],
            }
        persisted = self._read_persisted()
        if persisted and persisted.get("active"):
            return {
                "active": True,
                "applied_in_process": False,
                "unlocked_by": str(persisted.get("by") or "owner"),
                "unlocked_at": float(persisted.get("ts", 0.0)),
                "default_on": self._default_on(),
                "key_configured": bool(str(getattr(self.context.settings.partner, "owner_key", "") or ""))
                or seal_configured(),
                "changes": [],
            }
        return {
            "active": False,
            "applied_in_process": False,
            "unlocked_by": "",
            "unlocked_at": 0.0,
            "default_on": self._default_on(),
            "explicitly_locked": self._persisted_locked(),
            "limited_profile": self.limited_profile(),
            "key_configured": bool(str(getattr(self.context.settings.partner, "owner_key", "") or ""))
                or seal_configured(),
            "changes": [],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.status()

    # ── the dials ────────────────────────────────────────────────────────────
    def _node(self, section: str) -> Any:
        if section == "partner":
            return self.context.settings.partner
        return self.context.settings

    def _set(self, field: str, value: Any, section: str = "partner") -> PowerChanges | None:
        node = self._node(section)
        before = getattr(node, field, None)
        if before == value:
            return None
        setattr(node, field, value)
        return PowerChanges(field=field, before=before, after=value,
                            section=section)

    def _local_model_present(self) -> bool:
        """True when the owner has a model of their own downloaded."""
        try:
            if str(getattr(self.context.settings.llm, "local_model", "") or ""):
                return True
            router = getattr(self.context, "router", None)
            if router is not None:
                names = list(getattr(router, "providers", []) or [])
                if any(n.startswith("llama_cpp") for n in names):
                    return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def _widen(self) -> list[PowerChanges]:
        partner = self.context.settings.partner
        changes: list[PowerChanges] = []
        # Power mode = no volume caps at all (0 = unlimited).  The owner's
        # intent: normal mode keeps the protective defaults; *power* removes
        # every restriction except the human-like pacing (typing indicator,
        # per-chat spacing, quiet hours — those live elsewhere and stay).
        for field, value in (
            ("autonomy_mode", "auto"),
            ("max_parallel_chats", max(int(partner.max_parallel_chats), 8)),
            ("max_proactive_dm_per_day", 0),
            ("max_group_posts_per_day", 0),
            ("history_window", max(int(partner.history_window), 40)),
            ("typing_seconds", 0.6),
        ):
            change = self._set(field, value)
            if change is not None:
                changes.append(change)
        # The autonomy dial (wave 51): power mode cascades the whole
        # autonomous stack on — the cognitive loop (goals + projects +
        # improvement + personal fine-tune), the improvement auto-tick, and
        # the multi-model router (only when the owner has downloaded their
        # own local model, per the standing constraint).
        for field, value in (
            ("autonomy", replace(getattr(self.context.settings, "autonomy", None)
                                 or AutonomySettings(), enabled=True)),
            ("improvement", replace(getattr(self.context.settings,
                                            "improvement", None)
                                    or ImprovementSettings(),
                                    auto_tick=True)),
        ):
            change = self._set(field, value, section="root")
            if change is not None:
                changes.append(change)
        if self._local_model_present():
            change = self._set("router_intelligent", "on", section="root")
            if change is not None:
                changes.append(change)
        return changes

    def _restore(self) -> None:
        for change in reversed(self._restored):
            node = self._node(change.section)
            setattr(node, change.field, change.before)
        self._restored = []


def power_mode_for(context: Any) -> PowerMode:
    """Fetch (or create) the runtime's PowerMode instance.

    This is the choke point every power-mode consultation goes through, so
    the default-on posture is honored here: the first consultation
    auto-activates unless the owner explicitly locked power mode or turned
    the default off.  The auto-activation is audited.
    """
    power = context.extras.get("power")
    if not isinstance(power, PowerMode):
        power = PowerMode(context)
        context.extras["power"] = power
    if not power.active:
        # Restore a previously unlocked session first; otherwise honor the
        # default-on posture (explicit locks always win).
        if not power.adopt_persisted_state():
            power._auto_activate()
    return power
