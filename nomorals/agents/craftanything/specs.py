"""Declared effects for the tools that carry them.

Devon's ``ToolSpec`` gains an ``effects`` field (``None`` = never
declared); this table is the explicit declaration for the tools whose
effects the kind/capability derivation would misstate. Each entry names
its reason: the media organs' tools declare the union of their
actions' effects, and each action declares only what that action does
— ``inspect`` is read-only even on a tool that can write, so a call
that asks for more than its action declared is refused with the
missing effect named.

Tools absent from this table are derived by :func:`effects.derive_effects`
from their kind + capability. The table only ever *narrows* or *names*
what derivation cannot see; it never invents an effect a tool does not
have. An entry for a tool that is not in the registry (``anicraft``
before it lands) is inert — the engine applies the table only to tools
the live registry actually holds.
"""

from __future__ import annotations

from . import effects as E

__all__ = ["declared_effects", "action_effects", "EFFECTS_TABLE"]

#: Read-only actions: they open a document and report on it.
_READ_ONLY_ACTIONS = frozenset({"inspect", "parity", "attribution",
                                "library", "preset", "list_presets",
                                "open", "evaluate", "play",
                                "suggest_motion"})
#: Actions that change the document in memory and write it out.
_MUTATE_ACTIONS = frozenset({
    "create", "add", "set", "move", "rotate", "scale", "constrain",
    "solve", "dimension", "draft", "connect", "style", "text",
    "component", "instance", "prototype", "group", "layout", "guides",
    "transform", "boolean", "replace", "table", "metadata", "formula",
    "recalc", "notes", "reorder", "remove", "image", "mix", "automate",
    "plugin", "cook", "paint", "fill", "pattern", "blend", "compose",
    "mipmap", "normal", "dilate", "filter", "resize", "undo", "redo",
    "collab", "edit_keyframe", "retime", "build_preset", "save",
    "xfl_import", "import",
})
#: Actions that only write a rendered/exported artifact, not the document.
_WRITE_ACTIONS = frozenset({"render", "render_frame", "export",
                            "publish", "xfl_export"})


def _media_tool(actions, *, write=True, extra=None) -> dict:
    """One media organ tool: declared = union of its actions' effects."""
    declared = E.MUTATES_TARGET | E.IDEMPOTENT
    if write:
        declared |= E.WRITES_FILESYSTEM
    per_action: dict[str, int] = {}
    for action in actions:
        if action in _READ_ONLY_ACTIONS:
            per_action[action] = E.READ_ONLY
        elif action in _WRITE_ACTIONS:
            per_action[action] = E.WRITES_FILESYSTEM
        elif action in _MUTATE_ACTIONS:
            per_action[action] = E.MUTATES_TARGET | E.WRITES_FILESYSTEM
        else:  # pragma: no cover - every action above is classified
            per_action[action] = declared
    for action, mask in (extra or {}).items():
        per_action[action] = mask
        declared |= mask
    return {"effects": declared, "actions": per_action,
            "reason": "media organ tool: reads are read-only, edits "
                      "mutate the document and write it out"}


#: tool name -> {"effects": declared mask, "actions": {action: mask},
#:                "reason": str}. Only tools whose derivation would lie
#: are listed; everything else derives from kind + capability.
EFFECTS_TABLE: dict[str, dict] = {
    # The eight media organs (cadcraft's actions verified against
    # nomorals/tools/cadcraft.py _ACTIONS; the rest against their own
    # _ACTIONS tuples the same way).
    "cadcraft": _media_tool(
        ("inspect", "create", "add", "move", "rotate", "scale",
         "constrain", "solve", "dimension", "draft", "render",
         "import", "export", "parity", "attribution")),
    "materialcraft": _media_tool(
        ("inspect", "create", "add", "connect", "set", "cook",
         "render", "library", "preset", "export", "import",
         "parity", "attribution")),
    "texturecraft": _media_tool(
        ("inspect", "create", "add", "paint", "fill", "pattern",
         "blend", "compose", "render", "mipmap", "normal", "dilate",
         "filter", "resize", "undo", "parity", "attribution")),
    "anicraft": _media_tool(
        ("open", "save", "inspect", "evaluate", "render_frame", "play",
         "publish", "list_presets", "build_preset", "edit_keyframe",
         "retime", "undo", "redo", "xfl_import", "xfl_export",
         "suggest_motion")),
    "figcraft": _media_tool(
        ("inspect", "create", "add", "set", "style", "text",
         "component", "instance", "prototype", "render", "collab",
         "parity", "attribution")),
    "soundcraft": _media_tool(
        ("inspect", "create", "add", "set", "mix", "automate",
         "plugin", "render", "realtime", "parity", "attribution"),
        # realtime starts the realtime engine loop in-process: it
        # launches work, it does not write a file.
        extra={"realtime": E.LAUNCHES_PROCESS}),
    "designcraft": _media_tool(
        ("inspect", "create", "add", "style", "move", "resize",
         "group", "layout", "render", "guides", "parity",
         "attribution")),
    "vectorcraft": _media_tool(
        ("inspect", "create", "add", "style", "transform",
         "group", "boolean", "render", "parity", "attribution")),
    # Code execution runs arbitrary code: consent class, upstream's
    # EXECUTES_FOREIGN_CODE, exactly as its capability already says.
    "shell_run": {
        "effects": E.LAUNCHES_PROCESS | E.EXECUTES_FOREIGN_CODE,
        "actions": {},
        "reason": "runs a shell command: foreign code, needs consent"},
    "python_run": {
        "effects": E.LAUNCHES_PROCESS | E.EXECUTES_FOREIGN_CODE,
        "actions": {},
        "reason": "runs Python code: foreign code, needs consent"},
    "execute_plan": {
        "effects": E.LAUNCHES_PROCESS | E.EXECUTES_FOREIGN_CODE,
        "actions": {},
        "reason": "executes a plan of code steps: foreign code"},
    "code": {
        "effects": E.LAUNCHES_PROCESS | E.EXECUTES_FOREIGN_CODE,
        "actions": {},
        "reason": "runs code in the sandbox interpreter"},
    "code_session": {
        "effects": E.LAUNCHES_PROCESS | E.EXECUTES_FOREIGN_CODE,
        "actions": {},
        "reason": "runs code in a persistent sandbox session"},
    # Destructive filesystem work, named as such.
    "fs_delete": {
        "effects": (E.WRITES_FILESYSTEM | E.MUTATES_TARGET
                    | E.MAY_DISCARD_DATA),
        "actions": {},
        "reason": "deletes files: may discard data"},
    # Git push reaches the network and writes the remote.
    "git_push": {
        "effects": E.WRITES_FILESYSTEM | E.ACCESSES_NETWORK,
        "actions": {},
        "reason": "pushes commits to a remote over the network"},
    "git_commit": {
        "effects": E.WRITES_FILESYSTEM | E.MUTATES_TARGET,
        "actions": {},
        "reason": "writes commits to the local repository"},
}


def declared_effects(tool: str) -> int | None:
    """The table's declared effects for a tool, or ``None`` (derive)."""
    entry = EFFECTS_TABLE.get(tool)
    if entry is None:
        return None
    return entry["effects"]


def action_effects(tool: str, action: str) -> int | None:
    """The table's declared effects for one action, or ``None``."""
    entry = EFFECTS_TABLE.get(tool)
    if entry is None:
        return None
    return entry["actions"].get(action)
