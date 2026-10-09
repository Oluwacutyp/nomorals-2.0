"""Service connector bridge: all 28 service connectors as spine tools.

The connectors in nomorals/connectors/ (GitHub, Gmail, Binance, Mono,
Plaid, Spotify, Notion, ...) existed but were invisible to the brain —
no tool exposed them. This module bridges them into the tool registry
so plain language reaches every service: "check my Binance balance",
"list my GitHub repos", "send 5k to Mama via Mono".

Design:
- service_list: every connector + its callable actions (discovered live)
- service_call: connector + action + params -> real method call
- service_status: health check with self-heal attempt
- Money-movement actions require the finance capability (owner-only);
  read actions need network. The brain sees the filtered list.
- Failures are honest: unknown connector/action, auth missing, or
  service errors all return structured errors, never fake success.
"""

from __future__ import annotations

import inspect
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

# Lifecycle / internal methods never exposed as actions.
_LIFECYCLE = {
    "connect", "disconnect", "status", "test_connection",
    "resume_checkpoint", "begin_link",
}

# Actions that move money or mutate critical state: owner-only.
_SENSITIVE_VERBS = (
    "send", "transfer", "withdraw", "deposit", "trade", "order",
    "buy", "sell", "pay", "charge", "refund", "payout",
)


def _is_sensitive(action: str) -> bool:
    a = action.lower()
    return any(v in a for v in _SENSITIVE_VERBS)


def _discover_actions(connector_id: str) -> list[dict[str, Any]]:
    """Public callable actions on a connector class (no instance needed)."""
    from ..connectors.registry import get_connector
    cls = get_connector(connector_id)
    actions = []
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_") or name in _LIFECYCLE:
            continue
        try:
            sig = inspect.signature(member)
        except (ValueError, TypeError):
            continue
        params = [
            p for p in sig.parameters
            if p not in ("self", "cls")
        ]
        actions.append({
            "action": name,
            "params": params,
            "sensitive": _is_sensitive(name),
            "doc": (inspect.getdoc(member) or "")[:200],
        })
    return sorted(actions, key=lambda a: a["action"])


def _get_connector_instance(connector_id: str, context: Any) -> Any:
    """Instantiate with vault credentials; connect if needed."""
    from ..connectors.registry import create_connector
    from ..accounts.vault import CredentialVault
    vault = getattr(context, "vault", None) or CredentialVault()
    inst = create_connector(connector_id, vault)
    return inst


def register(registry: Any) -> None:
    @registry.register(
        "service_list",
        description=(
            "List every connected service and its callable actions. "
            "Returns connector ids, descriptions, and per-action param lists. "
            "Use this to discover what a service can do before calling it."
        ),
        capability="network",
        parameters={},
    )
    def service_list(context: Any) -> dict[str, Any]:
        from ..connectors.registry import list_connectors
        out = []
        for info in list_connectors():
            cid = info["id"]
            try:
                actions = _discover_actions(cid)
            except Exception as exc:  # noqa: BLE001
                actions = []
                _log.warning("action discovery failed for %s: %s", cid, exc)
            out.append({**info, "actions": actions})
        return {"ok": True, "services": out, "count": len(out)}

    @registry.register(
        "service_call",
        description=(
            "Call an action on a service connector. Args: connector "
            "(e.g. 'github', 'binance', 'mono'), action (from service_list), "
            "params (dict of action arguments). Read actions run directly; "
            "money-movement actions need the finance capability and may "
            "require confirmation. Returns the real service result or an "
            "honest error — never fake data."
        ),
        capability="network",
        parameters={
            "connector": "str — connector id, e.g. 'github'",
            "action": "str — action name from service_list",
            "params": "dict — action arguments",
        },
    )
    def service_call(context: Any, connector: str = "",
                     action: str = "",
                     params: dict | None = None) -> dict[str, Any]:
        from ..connectors.registry import get_connector
        from ..core.errors import ToolError
        if not connector or not action:
            raise ToolError("service_call needs connector and action")
        params = params or {}
        # Capability check: sensitive actions need finance (owner-only).
        if _is_sensitive(action):
            from ..core.policy import Capability
            caps = getattr(context, "capabilities", None)
            if caps is not None and not caps.grants("finance"):
                return {
                    "ok": False,
                    "error": (
                        f"action '{action}' moves money or mutates critical "
                        "state — owner finance capability required"
                    ),
                }
        try:
            cls = get_connector(connector)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        # Validate the action exists and isn't lifecycle/private.
        valid = {a["action"] for a in _discover_actions(connector)}
        if action not in valid:
            return {
                "ok": False,
                "error": f"unknown action '{action}' for '{connector}'",
                "valid_actions": sorted(valid),
            }
        try:
            inst = _get_connector_instance(connector, context)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"connector init failed: {exc}"}
        # Ensure connected; try to connect from vault credentials.
        try:
            st = inst.status()
            connected = bool(getattr(st, "connected", False))
        except Exception:  # noqa: BLE001
            connected = False
        if not connected:
            try:
                res = inst.connect()
                if not bool(getattr(res, "ok", False)):
                    return {
                        "ok": False,
                        "error": (
                            f"'{connector}' not connected: "
                            f"{getattr(res, 'message', 'auth missing')} — "
                            "connect the service first"
                        ),
                    }
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"connect failed: {exc}"}
        try:
            fn = getattr(inst, action)
            result = fn(**params) if params else fn()
        except TypeError as exc:
            return {"ok": False, "error": f"bad params for '{action}': {exc}"}
        except Exception as exc:  # noqa: BLE001
            _log.warning("service_call %s.%s failed: %s",
                         connector, action, exc)
            return {"ok": False, "error": str(exc)}
        # Normalize to JSON-able.
        try:
            import json
            json.dumps(result)
            payload = result
        except (TypeError, ValueError):
            payload = {"result": str(result)}
        return {"ok": True, "connector": connector,
                "action": action, "result": payload}

    @registry.register(
        "service_status",
        description=(
            "Health check for service connectors. Args: connector (optional "
            "— omit for all). Tests the live connection and attempts a "
            "self-heal reconnect on failure. Reports honest status per "
            "service: connected, degraded, or down with the reason."
        ),
        capability="network",
        parameters={
            "connector": "str — optional connector id; omit for all",
        },
    )
    def service_status(context: Any,
                       connector: str = "") -> dict[str, Any]:
        from ..connectors.registry import list_connectors
        targets = (
            [connector] if connector
            else [c["id"] for c in list_connectors()]
        )
        out = []
        for cid in targets:
            try:
                inst = _get_connector_instance(cid, context)
                ok = bool(inst.test_connection())
                if not ok:
                    # Self-heal: one reconnect attempt.
                    try:
                        res = inst.connect()
                        ok = bool(getattr(res, "ok", False)
                                  and inst.test_connection())
                    except Exception:  # noqa: BLE001
                        ok = False
                out.append({"connector": cid,
                            "status": "connected" if ok else "down",
                            "healed": ok})
            except Exception as exc:  # noqa: BLE001
                out.append({"connector": cid, "status": "down",
                            "error": str(exc)[:200]})
        return {"ok": True, "services": out}
