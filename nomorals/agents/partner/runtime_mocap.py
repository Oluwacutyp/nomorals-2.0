"""RuntimeMocapMixin: PartnerRuntime command group (mocap + rig)."""

from __future__ import annotations

import json
from typing import Any

from ...core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)


class RuntimeMocapMixin:
    """RuntimeMocapMixin for :class:`PartnerRuntime`."""

    # ── mocap + rig: /mocap ─────────────────────────────────────────────
    def _control_mocap(self, tail: str, chat_key: str,
                       message: Any = None) -> str:
        """Motion capture and the VTuber rig, live from any chat.

        /mocap status                             store channels + engines
        /mocap rig list                           loaded rig sessions
        /mocap rig load [name]                    load a rig (default rig)
        /mocap rig status [name]                  a session's rig + remap
        /mocap rig pose [name] [frame JSON]       drive one frame
        /mocap rig render [name]                  render the pose (PNG)
        /mocap rig drive <bvh> [text]             drive a take, render it
        /mocap rig export <bvh> [text]            drive a take, BVH back out
        """
        from ...media.mocap.bvh import SkeletonFrame
        from ...media.rig import (
            FaceInput,
            get_session,
            list_sessions,
            load_session,
            load_take_bvh,
        )
        from ...media.rig.errors import RigError
        from ...tools.mocap import _STORE

        parts = (tail or "").split()
        if not parts:
            return self._control_mocap_help()
        if parts[0] == "status":
            return ("mocap channels: "
                    + ", ".join(_STORE.channels())
                    + " | engines: "
                    + ", ".join(e["name"] for e in _STORE.engines()))
        if parts[0] != "rig":
            return self._control_mocap_help()
        verb = parts[1] if len(parts) > 1 else ""
        name = "default"
        rest = parts[2:]
        if verb in ("load", "status", "pose", "render") and rest \
                and not rest[0].startswith("{"):
            name = rest[0]
            rest = rest[1:]
        try:
            if verb == "list":
                sessions = list_sessions()
                lines = ["rig sessions:"]
                for entry in sessions:
                    lines.append(
                        f"  {entry['name']}: {entry['rig']} "
                        f"({entry['bones']} bones, "
                        f"remap {entry['remap_dialect']}, "
                        f"{entry['width']}x{entry['height']} "
                        f"@{entry['fps']}fps)")
                return "\n".join(lines)
            if verb == "load":
                session = load_session(name)
                return (f"rig {session.rig.name} loaded as {name} "
                        f"({len(session.rig.bones)} bones, remap "
                        f"{session.remap.dialect})")
            if verb == "status":
                session = get_session(name)
                info = dict({"name": name}, **session.to_dict())
                return json.dumps(info, indent=2)
            session = get_session(name)
            if verb == "pose":
                face = None
                frame = None
                for token in rest:
                    if token.startswith("{"):
                        try:
                            raw = json.loads(token)
                        except ValueError:
                            return "pose frame/face must be JSON"
                        if isinstance(raw, dict) and (
                                "blendshapes" in raw
                                or "viseme" in raw
                                or "text" in raw):
                            face = FaceInput(
                                blendshapes=raw.get("blendshapes"),
                                viseme=raw.get("viseme"),
                                text=raw.get("text"),
                                laughter=float(
                                    raw.get("laughter") or 0.0))
                        elif isinstance(raw, dict):
                            frame = SkeletonFrame(values={
                                str(k): tuple(float(v) for v in vals)
                                for k, vals in raw.items()})
                pose = session.drive_frame(frame, face)
                return (f"pose f:{pose.seq} viseme {pose.viseme} "
                        f"mouth={tuple(round(v, 2) for v in pose.mouth)}"
                        + (f" unmapped: {', '.join(pose.unmapped)}"
                           if pose.unmapped else "")
                        + (" STALE" if pose.stale else ""))
            if verb == "render":
                pose = session.drive_frame(None)
                rgb = session.render_frame(pose)
                return (f"pose f:{pose.seq} rendered "
                        f"({len(rgb)} bytes RGB, "
                        f"{session.width}x{session.height}) — "
                        f"use nm mocap rig render --out PATH.png "
                        f"for the file")
            if verb in ("drive", "export"):
                if not rest:
                    return f"/mocap rig {verb} needs a BVH path"
                take = load_take_bvh(rest[0])
                text = " ".join(rest[1:]) or None
                posed = session.drive_take(
                    take["skeleton"], take["frames"], text=text)
                if verb == "export":
                    result = session.export_bvh(
                        posed, f"rig-drive-{name}.bvh", store=_STORE)
                    return (f"exported {result['frames']} frames to "
                            f"{result['path']}")
                return (f"drove {len(posed)} frames "
                        f"(visemes: "
                        f"{', '.join(p.viseme for p in posed[:8])}"
                        f"{'…' if len(posed) > 8 else ''})")
        except RigError as exc:
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_mocap", exc)
            return f"rig: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat never dies
            record_failure(
                "runtime_mocap.RuntimeMocapMixin._control_mocap", exc)
            return f"mocap failed: {exc}"
        return self._control_mocap_help()

    def _control_mocap_help(self) -> str:
        return (
            "/mocap status | rig list | rig load [name] | "
            "rig status [name] | rig pose [name] [frame JSON] | "
            "rig render [name] | rig drive <bvh> [text] | "
            "rig export <bvh> [text]")
