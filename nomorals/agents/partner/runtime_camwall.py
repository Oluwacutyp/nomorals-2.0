"""RuntimeCamwallMixin: PartnerRuntime command group (cameras wall)."""

from __future__ import annotations

from typing import Any

from ...core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)


class RuntimeCamwallMixin:
    """RuntimeCamwallMixin for :class:`PartnerRuntime`."""

    # ── cameras wall: /camwall ────────────────────────────────────────
    def _control_camwall(self, tail: str, chat_key: str,
                         message: Any = None) -> str:
        """IP cameras wall — constantly refreshing camera feeds.

        /camwall load [config] [rows N] [server local|remote]
                                        load a wall from a config
        /camwall list                   walls loaded now
        /camwall start [wall]           start a wall's loop
        /camwall stop [wall]            stop a wall's loop
        /camwall status [wall]          one wall's cameras + frames
        /camwall tick [wall]            one loop pass
        /camwall pause <camera> [wall]  pause one camera
        /camwall resume <camera> [wall] resume one camera
        /camwall focus <camera> [wall]  singleview one camera
        /camwall wall [wall]            back to the whole wall
        /camwall server local|remote [wall]
                                        which host each camera uses
        /camwall rows <1-6> [wall]      images per row
        /camwall snapshot <camera> <path> [wall]
                                        save a camera's latest frame
        /camwall export <dir> [wall]    the wall as a page
        /camwall serve <dir>            serve an exported wall
        /camwall unserve <dir>          stop serving it
        /camwall served                 servers running now
        """
        from ...tools.camwall import camwall_sync

        parts = (tail or "").split()
        if not parts:
            return self._control_camwall_help()
        verb = parts[0]
        rest = parts[1:]
        try:
            if verb == "list":
                result = camwall_sync(action="list")
                walls = result.get("walls", [])
                if not walls:
                    return ("camwall: no walls loaded "
                            "(/camwall load <config> first)")
                lines = ["camwall walls:"]
                for entry in walls:
                    lines.append(
                        f"  {entry.get('wall')}: "
                        f"{entry.get('cameras')} cameras, "
                        f"{'running' if entry.get('running') else 'stopped'}")
                return "\n".join(lines)
            if verb == "served":
                result = camwall_sync(action="served")
                servers = result.get("servers", [])
                if not servers:
                    return "camwall: no walls served"
                lines = ["camwall served:"]
                for server in servers:
                    lines.append(f"  {server.get('dir')} — "
                                 f"{server.get('url')}")
                return "\n".join(lines)
            if verb == "load":
                config = rest[0] if rest else None
                rows = 2
                server = "remote"
                for item in rest[1:]:
                    if item == "local":
                        server = "local"
                    elif item == "remote":
                        server = "remote"
                    elif item.isdigit():
                        rows = int(item)
                result = camwall_sync(
                    action="load",
                    params={"config": config, "rows": rows,
                            "server": server})
                if not result.get("ok"):
                    return f"camwall: {result.get('error')}"
                return self._camwall_status_text(result)
            if verb in ("status", "start", "stop", "drop", "tick", "wall"):
                wall = rest[0] if rest else "default"
                result = camwall_sync(action=verb, node=wall)
                if not result.get("ok"):
                    return f"camwall: {result.get('error')}"
                if verb == "status":
                    return self._camwall_status_text(result)
                if verb == "tick":
                    return (f"camwall: ticked "
                            f"{len(result.get('ticked', []))} camera(s)")
                return f"camwall: wall {wall} {verb} ok"
            if verb in ("pause", "resume", "focus"):
                if not rest:
                    return self._control_camwall_help()
                camera = rest[0]
                wall = rest[1] if len(rest) > 1 else "default"
                result = camwall_sync(action=verb, node=wall, value=camera)
                if not result.get("ok"):
                    return f"camwall: {result.get('error')}"
                return f"camwall: camera {camera} {verb} ok"
            if verb == "server":
                if not rest or rest[0] not in ("local", "remote"):
                    return self._control_camwall_help()
                wall = rest[1] if len(rest) > 1 else "default"
                result = camwall_sync(action="server", node=wall,
                                      value=rest[0])
                if not result.get("ok"):
                    return f"camwall: {result.get('error')}"
                return f"camwall: wall {wall} server {result.get('server')}"
            if verb == "rows":
                if not rest or not rest[0].isdigit():
                    return self._control_camwall_help()
                wall = rest[1] if len(rest) > 1 else "default"
                result = camwall_sync(action="rows", node=wall,
                                      value=int(rest[0]))
                if not result.get("ok"):
                    return f"camwall: {result.get('error')}"
                return (f"camwall: wall {wall} "
                        f"{result.get('rows')} per row")
            if verb == "snapshot":
                if len(rest) < 2:
                    return self._control_camwall_help()
                camera, output = rest[0], rest[1]
                wall = rest[2] if len(rest) > 2 else "default"
                result = camwall_sync(action="snapshot", node=wall,
                                      value=camera, output=output)
                if not result.get("ok"):
                    return f"camwall: {result.get('error')}"
                return (f"camwall: saved {result.get('bytes')} bytes "
                        f"to {result.get('output')}")
            if verb == "export":
                if not rest:
                    return self._control_camwall_help()
                output = rest[0]
                wall = rest[1] if len(rest) > 1 else "default"
                result = camwall_sync(action="export", node=wall,
                                      output=output)
                if not result.get("ok"):
                    return f"camwall: {result.get('error')}"
                return (f"camwall: wall {wall} exported to "
                        f"{result.get('dir')} "
                        f"({result.get('cameras')} cameras, "
                        f"{result.get('frames')} frames)")
            if verb == "serve":
                if not rest:
                    return self._control_camwall_help()
                result = camwall_sync(action="serve", output=rest[0])
                if not result.get("ok"):
                    return f"camwall: {result.get('error')}"
                return (f"camwall: serving {result.get('dir')} at "
                        f"{result.get('url')}")
            if verb == "unserve":
                if not rest:
                    return self._control_camwall_help()
                result = camwall_sync(action="unserve", output=rest[0])
                return f"camwall: stopped serving {result.get('dir')}"
            return self._control_camwall_help()
        except Exception as exc:  # noqa: BLE001 - chat replies, never traces
            record_failure("runtime_camwall._control_camwall", exc)
            _log.exception("camwall command failed: %s", exc)
            return f"camwall: failed ({exc})"

    def _control_camwall_help(self) -> str:
        return (
            "camwall — IP cameras wall:\n"
            "/camwall load [config] [rows N] [server local|remote]\n"
            "/camwall list | status [wall] | start [wall] | stop [wall]\n"
            "/camwall tick [wall] | pause <camera> | resume <camera>\n"
            "/camwall focus <camera> | wall | server local|remote\n"
            "/camwall rows <1-6> | snapshot <camera> <path>\n"
            "/camwall export <dir> | serve <dir> | unserve <dir> | served")

    def _camwall_status_text(self, status: dict[str, Any]) -> str:
        lines = [f"wall {status.get('wall')}: "
                 f"{'running' if status.get('running') else 'stopped'} "
                 f"({status.get('server')} server, "
                 f"{status.get('rows')} per row)"]
        for cam in status.get("camera_list", []):
            state = "paused" if cam.get("paused") else "live"
            frame = "frame" if cam.get("has_frame") else "no frame"
            error = (f" — {cam.get('last_error')}"
                     if cam.get("last_error") else "")
            lines.append(
                f"  [{cam.get('id')}] {cam.get('name')} — "
                f"{cam.get('location')} — {state} — {frame} — "
                f"{cam.get('fps')} fps — "
                f"{cam.get('counts', {}).get('load', 0)} loads, "
                f"{cam.get('counts', {}).get('fail', 0)} fails{error}")
        return "\n".join(lines)
