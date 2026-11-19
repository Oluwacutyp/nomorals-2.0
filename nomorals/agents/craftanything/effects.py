"""Declared side effects of a tool.

Port of upstream ``Effects`` (``crates/engine/src/tool.rs``): every tool
declares what a call can touch; MCP hints and consent are derived from
the declaration, never guessed per call. The bit layout is upstream's
verbatim (bits 0–10); bits 11–12 are Devon extensions for the two
effect classes upstream has no flag for — sending a message and
spending money — and are named as such wherever they appear.

Effects are plain ints (a bitmask), not an enum: the registry stores
them on ``ToolSpec.effects`` as an int (``None`` = never declared —
the three-state habit from ``lease_needs``), and this module derives
them from a spec's kind/capability/annotations when nothing was
declared, and infers what one particular call asks for from its
parameters. A call whose requested effects are not a subset of the
declared effects is refused with the missing effect named, before
anything executes.
"""

from __future__ import annotations

__all__ = [
    "MUTATES_SESSION", "MUTATES_TARGET", "WRITES_FILESYSTEM",
    "LAUNCHES_PROCESS", "ACCESSES_NETWORK", "CHANGES_UI_STATE",
    "MAY_DISCARD_DATA", "EXECUTES_FOREIGN_CODE", "MUTATES_DEVICE",
    "IDEMPOTENT", "CARRIES_TARGET_CONTENT",
    "SEND_MESSAGE", "SPENDS_MONEY",
    "EVIDENCE", "NATIVE_EVIDENCE", "NETWORK_EVIDENCE", "READ_ONLY",
    "EFFECT_NAMES", "effect_names", "effects_from_names",
    "read_only_hint", "destructive_hint", "open_world_hint",
    "needs_consent", "derive_effects", "infer_requested_effects",
]

#: Records evidence or changes session state (almost every analysis tool).
MUTATES_SESSION = 1 << 0
#: Changes the target or its annotations (rename, comment, edit a document).
MUTATES_TARGET = 1 << 1
WRITES_FILESYSTEM = 1 << 2
LAUNCHES_PROCESS = 1 << 3
ACCESSES_NETWORK = 1 << 4
CHANGES_UI_STATE = 1 << 5
MAY_DISCARD_DATA = 1 << 6
#: Runs code from the target or a script. Needs consent.
EXECUTES_FOREIGN_CODE = 1 << 7
#: Changes a connected device (install, push, start). Needs consent.
MUTATES_DEVICE = 1 << 8
IDEMPOTENT = 1 << 9
#: Its result carries content taken from the target: clients are told to
#: treat it as untrusted data.
CARRIES_TARGET_CONTENT = 1 << 10
#: Sends a message to a person or a group (Devon extension: upstream has
#: no send flag; the brief's effect classes name send explicitly).
SEND_MESSAGE = 1 << 11
#: Spends the owner's money (Devon extension, same reason as send).
SPENDS_MONEY = 1 << 12

#: Plain evidence-producing analysis.
EVIDENCE = MUTATES_SESSION | IDEMPOTENT
#: Evidence that needs an engine process.
NATIVE_EVIDENCE = EVIDENCE | LAUNCHES_PROCESS
#: Evidence that talks to a browser or device over the network.
NETWORK_EVIDENCE = EVIDENCE | ACCESSES_NETWORK
#: Pure reads with no effect at all.
READ_ONLY = IDEMPOTENT

EFFECT_NAMES: dict[int, str] = {
    MUTATES_SESSION: "mutates_session",
    MUTATES_TARGET: "mutates_target",
    WRITES_FILESYSTEM: "writes_filesystem",
    LAUNCHES_PROCESS: "launches_process",
    ACCESSES_NETWORK: "accesses_network",
    CHANGES_UI_STATE: "changes_ui_state",
    MAY_DISCARD_DATA: "may_discard_data",
    EXECUTES_FOREIGN_CODE: "executes_foreign_code",
    MUTATES_DEVICE: "mutates_device",
    IDEMPOTENT: "idempotent",
    CARRIES_TARGET_CONTENT: "carries_target_content",
    SEND_MESSAGE: "send_message",
    SPENDS_MONEY: "spends_money",
}

_NAME_TO_EFFECT = {name: bit for bit, name in EFFECT_NAMES.items()}


def effect_names(effects: int) -> list[str]:
    """Flag names in bit order; unknown bits are named, never dropped."""
    names = [name for bit, name in sorted(EFFECT_NAMES.items())
             if effects & bit]
    unknown = effects & ~sum(EFFECT_NAMES)
    if unknown:
        names.append(f"unknown_0x{unknown:x}")
    return names


def effects_from_names(names) -> int:
    """Inverse of :func:`effect_names`; unknown names raise ValueError."""
    mask = 0
    for name in names:
        try:
            mask |= _NAME_TO_EFFECT[name]
        except KeyError:
            raise ValueError(f"unknown effect name {name!r}") from None
    return mask


def read_only_hint(effects: int) -> bool:
    """True when the tool does nothing but repeatable reads (upstream)."""
    return (effects & ~IDEMPOTENT) == 0


def destructive_hint(effects: int) -> bool:
    """True when the tool can destroy data, run foreign code, or mutate a device."""
    return bool(effects & (MAY_DISCARD_DATA | EXECUTES_FOREIGN_CODE
                           | MUTATES_DEVICE))


def open_world_hint(effects: int) -> bool:
    """True when the tool launches processes or touches the network."""
    return bool(effects & (LAUNCHES_PROCESS | ACCESSES_NETWORK))


def needs_consent(effects: int) -> bool:
    """True when the effects need the owner's OK first.

    Upstream's rule is exactly ``EXECUTES_FOREIGN_CODE | MUTATES_DEVICE``;
    Devon adds ``SPENDS_MONEY`` — spending the owner's money is consent
    class by the same standing rule that makes installs consent class.
    """
    return bool(effects & (EXECUTES_FOREIGN_CODE | MUTATES_DEVICE
                           | SPENDS_MONEY))


#: Capability strings (``nomorals.core.policy.Capability``) mapped to the
#: effects they imply. A tool's derived effects are the union of its
#: kind's effects and its capability's effects — an over-approximation
#: on purpose: derivation must never under-claim what a tool can do.
_CAPABILITY_EFFECTS: dict[str, int] = {
    "fs.read": IDEMPOTENT,
    "fs.write": WRITES_FILESYSTEM | MUTATES_TARGET,
    "fs.delete": WRITES_FILESYSTEM | MUTATES_TARGET | MAY_DISCARD_DATA,
    "exec.shell": LAUNCHES_PROCESS,
    "exec.code": LAUNCHES_PROCESS | EXECUTES_FOREIGN_CODE,
    "exec.install": LAUNCHES_PROCESS | MUTATES_DEVICE,
    "net.out": ACCESSES_NETWORK,
    "net.download": ACCESSES_NETWORK | WRITES_FILESYSTEM,
    "net.browser": ACCESSES_NETWORK,
    "model.call": ACCESSES_NETWORK,
    "model.download": ACCESSES_NETWORK | WRITES_FILESYSTEM,
    "train.run": LAUNCHES_PROCESS,
    "train.gpu": LAUNCHES_PROCESS,
    "mem.read": IDEMPOTENT,
    "mem.write": MUTATES_SESSION,
    "db.read": IDEMPOTENT,
    "db.write": MUTATES_SESSION | WRITES_FILESYSTEM,
    "db.admin": MUTATES_SESSION | WRITES_FILESYSTEM | MAY_DISCARD_DATA,
    "agent.spawn": LAUNCHES_PROCESS | MUTATES_SESSION,
    "mission.start": LAUNCHES_PROCESS | MUTATES_SESSION,
    "media.gen": WRITES_FILESYSTEM | MUTATES_TARGET,
    "social.read": ACCESSES_NETWORK | IDEMPOTENT,
    "social.post": SEND_MESSAGE | ACCESSES_NETWORK,
    "social.dm": SEND_MESSAGE | ACCESSES_NETWORK,
    "social.bulk": SEND_MESSAGE | ACCESSES_NETWORK,
    "social.play": ACCESSES_NETWORK | IDEMPOTENT,
    "sys.config": MUTATES_SESSION,
    "sys.backup": WRITES_FILESYSTEM,
    "sys.shutdown": MUTATES_DEVICE | MAY_DISCARD_DATA,
}

#: Kinds (``ToolSpec.kind``) mapped to their effects. The ``io`` default
#: contributes nothing — an undeclared ``io`` tool's effects come from
#: its capability, or from the explicit table in :mod:`specs`.
_KIND_EFFECTS: dict[str, int] = {
    "query": READ_ONLY,
    "read": READ_ONLY,
    "search": READ_ONLY,
    "list": READ_ONLY,
    "write": WRITES_FILESYSTEM | MUTATES_TARGET,
    "delete": WRITES_FILESYSTEM | MUTATES_TARGET | MAY_DISCARD_DATA,
    "exec": LAUNCHES_PROCESS,
    "net": ACCESSES_NETWORK | IDEMPOTENT,
    "web": ACCESSES_NETWORK | IDEMPOTENT,
    "external": ACCESSES_NETWORK | IDEMPOTENT,
}


def derive_effects(kind: str, capability: str,
                   annotations: dict | None = None) -> int:
    """Derive a tool's effects from its kind, capability and annotations.

    Deliberately generous: a derived set that under-claims would let a
    call through that the declaration should have caught. Tools with a
    real declaration (``ToolSpec.effects`` or the table in
    :mod:`specs`) never reach this function.
    """
    effects = _KIND_EFFECTS.get((kind or "").lower(), 0)
    if capability:
        effects |= _CAPABILITY_EFFECTS.get(capability, 0)
    annotations = annotations or {}
    if annotations.get("destructive"):
        effects |= MAY_DISCARD_DATA
    return effects


#: Parameter names whose presence means the call writes a file. The
#: craft tools route every write through ``output``; other tools use
#: ``path``/``target``/``dest`` the same way.
_WRITE_PARAMS = ("output", "output_path", "target", "dest", "destination",
                 "save_to", "write_to")
#: Parameter names whose presence means the call reaches the network.
_NETWORK_PARAMS = ("url", "webhook", "endpoint")


def infer_requested_effects(action: str | None,
                            params: dict | None) -> int:
    """What one particular call asks for, from its action and parameters.

    Only the call's own footprint: an ``output`` path means a filesystem
    write, a ``url`` means network access. The tool's declared effects
    are checked against this — a call that asks for more than the tool
    declared is refused with the missing effect named.
    """
    params = params or {}
    requested = 0
    for key in _WRITE_PARAMS:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            requested |= WRITES_FILESYSTEM
            break
    for key in _NETWORK_PARAMS:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            requested |= ACCESSES_NETWORK
            break
    return requested
