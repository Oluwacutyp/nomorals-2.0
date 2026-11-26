"""Group cleanup as a brain tool: ``group_cleanup`` — the brain's
answer to "clean up the deleted accounts in <group>" and "remove
the inactive members".

One tool, action-dispatched, over the runtime's own command
methods (``_control_cleanup`` / ``_control_rmdeleted`` /
``_control_rminactive``) — the tool is only the surface; the
removal engine, the standing re-checks, the journaling and the
owner-stop gate live in exactly one place, the runtime methods
the chat commands use.  No second removal path exists.

``action=scan`` reports the cleanup scan (deleted accounts +
inactive past the window + the members it cannot judge — a dry
run by construction, it removes nothing).  ``action=rmdeleted``
removes the deleted accounts the scan finds.  ``action=rminactive``
removes members inactive past the window (``days=`` overrides
this run).  ``action=status`` reports the group's cleanup
settings, the last automatic pass and whether the automatic pass
is currently held (owner stop / degraded brain) — a pure surface
read, no runtime needed.

The runtime is resolved from ``context.extras["partner_runtime"]``
(the live PartnerRuntime publishes itself there at boot, next to
the gateway and the competence gate) — never a second wire, never
an import of the runtime from the tool layer.  Without a live
runtime the tool refuses in words: the scan needs the runtime's
member enumeration, the removals need its admin seam.  The
message the tool hands the runtime carries the gateway's own
owner stamp (``meta={"is_owner": True}``) — the tool only ever
runs for the owner, and the stamp is exactly what the gateway
stamps on his real commands.  The group is resolved by name or
key through the same matcher the ``group_health`` tool uses; an
unknown or ambiguous group is answered honestly, never guessed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ..core.policy import Capability
from .group_health import _find_groups

__all__ = ["register"]

_ACTIONS = ("scan", "rmdeleted", "rminactive", "status")


def _surface(context: Any) -> Any:
    from ..social.chat.group_surface import default_surface

    settings = getattr(context, "settings", None)
    ws_dir = getattr(settings, "workspace_dir", ".") or "."
    return default_surface(ws_dir)


def _runtime(context: Any) -> Any | None:
    extras = getattr(context, "extras", None)
    if isinstance(extras, dict):
        return extras.get("partner_runtime")
    return getattr(context, "partner_runtime", None)


def _resolve_group(surface: Any, group: str) -> tuple[str, str]:
    """(key, error): the one group ``group`` names, or the
    plain-word refusal.  A bare key is used as-is when the
    surface knows it; a name must match exactly one group."""
    needle = (group or "").strip()
    if not needle:
        return "", "which group? give its name or key"
    # A bare key is used as-is when the surface knows it — read
    # ``data`` directly, never ``store.group()``: that call
    # MATERIALISES an unknown key (a probe that creates phantom
    # groups is how the site publish path once died mid-iteration).
    store = getattr(surface, "store", None)
    data = getattr(store, "data", None) or {}
    if needle in data:
        return needle, ""
    matches = _find_groups(surface, needle)
    if len(matches) == 1:
        return matches[0], ""
    if not matches:
        return "", f"no group on record matches {needle!r}"
    return "", (f"{needle!r} matches {len(matches)} groups — "
                "name one exactly")


def _owner_message(key: str, surface: Any) -> Any:
    """The message the runtime's command methods expect, as the
    owner: the group chat ref plus the gateway's owner stamp.
    The runtime's operator check reads the stamp first, so the
    tool acts with the owner's authority and nothing else's."""
    platform, _, chat_id = key.partition(":")
    entry = {}
    try:
        entry = surface.store.group(key)
    except Exception:  # noqa: BLE001 - title is garnish
        entry = {}
    title = str(entry.get("title") or entry.get("name") or "")
    chat = SimpleNamespace(platform=platform, key=key,
                           kind="group", title=title,
                           chat_id=chat_id)
    return SimpleNamespace(chat=chat, sender_id="", sender="owner",
                           sender_username="", text="", meta={
                               "is_owner": True},
                           reply_to="", message_id="")


def _status_text(surface: Any, key: str) -> str:
    cfg = surface.cleanup_cfg(key)
    entry = surface.store.group(key)
    auto = entry.get("cleanup_auto") or {}
    held = (entry.get("extras") or {}).get("cleanup_auto_held") or {}
    lines = [
        f"cleanup for {key}: removals "
        f"{'ON' if cfg.get('enabled', True) else 'OFF'} · "
        f"auto-remove deleted "
        f"{'on' if cfg.get('auto_deleted', True) else 'off'} · "
        f"inactivity window "
        f"{int(cfg.get('inactive_days') or 90)} days",
    ]
    last = auto.get("last_run")
    lines.append("last automatic pass: "
                 + (f"{float(last):.0f}" if last else "never"))
    if held:
        lines.append("automatic pass HELD: "
                     f"{held.get('reason') or 'held'}")
    return "\n".join(lines)


def register(registry: Any) -> None:
    """Attach the ``group_cleanup`` tool."""
    context = registry.context

    @registry.register(
        "group_cleanup",
        description=(
            "Telegram group member cleanup: scan a claimed group "
            "for deleted accounts and inactive members, and remove "
            "them. action=scan reports (dry run, removes nothing); "
            "action=rmdeleted kicks the deleted accounts the scan "
            "finds; action=rminactive kicks members inactive past "
            "the window (days= overrides this run); action=status "
            "reports the group's cleanup settings and whether the "
            "automatic pass is held. group=<name or key> names the "
            "group. Kicks, never bans; admins, the owner and "
            "members whose standing cannot be verified are never "
            "removed; every outcome is journaled. A stopped "
            "platform refuses removals; the scan still runs."
        ),
        capability=Capability.SOCIAL_BULK,
    )
    def group_cleanup_tool(
        action: str = "scan",
        group: str = "",
        days: str = "",
    ) -> dict[str, Any]:
        action = str(action or "scan").strip().lower()
        if action not in _ACTIONS:
            return {"ok": False,
                    "error": f"unknown action {action!r} — "
                             f"actions are: {', '.join(_ACTIONS)}"}
        surface = _surface(context)
        key, error = _resolve_group(surface, group)
        if error:
            return {"ok": False, "error": error}
        if action == "status":
            return {"ok": True, "text": _status_text(surface, key)}
        runtime = _runtime(context)
        if runtime is None:
            return {"ok": False,
                    "error": ("no live runtime in this process — "
                              "group cleanup runs where Devon is "
                              "running (the partner runtime); "
                              "start her and ask again")}
        message = _owner_message(key, surface)
        try:
            if action == "scan":
                text = runtime._control_cleanup("", message)
            elif action == "rmdeleted":
                text = runtime._control_rmdeleted("", message)
            else:
                days_txt = str(days or "").strip().lower()
                if days_txt:
                    digits = days_txt.rstrip("d").strip()
                    if not digits.isdigit():
                        return {"ok": False,
                                "error": "days must be a number"}
                    days_txt = digits
                text = runtime._control_rminactive(days_txt, message)
        except Exception as exc:  # noqa: BLE001 - verdict
            return {"ok": False, "error": str(exc)[:300]}
        return {"ok": True, "text": str(text or "")}
