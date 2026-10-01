"""Workspace tools (wave 84): the main AI's window onto the virtual CPU
farm — inspect every core, watch the fleet's load, scale it, pause or
resume individual cores."""
from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from ..workspace import Workspace


def _ws(context: Any) -> Workspace:
    ws = getattr(context, "workspace", None)
    if ws is None:
        ws = Workspace(context)
        setattr(context, "workspace", ws)
    return ws


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "workspace_status",
        description=(
            "The virtual CPU farm: environment profile (termux/mobile/pc/"
            "vps/workstation + its min-target-max VCPU envelope), every "
            "core's status (idle/busy/paused/error), load, queue depth, "
            "task stats, and autoscale state. Use before heavy parallel "
            "work to see how much capacity exists."
        ),
        capability=Capability.FS_READ,
        parameters={
            "verbose": "bool (optional, false) — include per-core detail",
        },
    )
    def workspace_status(*, verbose: str = "") -> dict[str, Any]:
        do = str(verbose or "").lower() in {"1", "true", "yes", "on"}
        ws = _ws(context)
        st = ws.status()
        if not do:
            st["vcpus_short"] = [
                {"id": d["id"], "status": d["status"], "load": d["load"],
                 "queue": d["queue"], "tasks": d["stats"]["tasks_run"]}
                for d in st["details"]]
            st = {k: v for k, v in st.items() if k != "details"}
        st["summary"] = ws.summary_line()
        return st

    @registry.register(
        "workspace_control",
        description=(
            "Control the virtual CPU farm. action=scale_to N resizes the "
            "pool (clamped to the profile envelope); scale_up/scale_down "
            "add or retire one core (retire drains its queue first); "
            "pause/resume <vcpu> freezes or unfreezes one core (queued "
            "work waits while paused); add <kind> attaches a dedicated "
            "io/cpu core; remove <vcpu> detaches one. status shows the "
            "farm."
        ),
        capability=Capability.FS_READ,
        parameters={
            "action": "str — status | scale_to | scale_up | scale_down | "
                      "pause | resume | add | remove",
            "value": "str — N for scale_to; vcpu name for pause/resume/"
                     "remove; core kind (io|cpu|balanced) for add",
        },
    )
    def workspace_control(*, action: str = "status",
                          value: str = "") -> dict[str, Any]:
        ws = _ws(context)
        action = (action or "status").strip().lower()
        value = (value or "").strip()
        if action == "status":
            st = ws.status()
            st.pop("details", None)
            st["summary"] = ws.summary_line()
            return st
        if action == "scale_to":
            try:
                n = int(value)
            except ValueError:
                return {"error": "scale_to needs a number, e.g. 6"}
            return {"vcpus": ws.scale_to(n), "summary": ws.summary_line()}
        if action == "scale_up":
            return {"vcpus": ws.scale_up(1), "summary": ws.summary_line()}
        if action == "scale_down":
            return {"vcpus": ws.scale_down(1), "summary": ws.summary_line()}
        if action in {"pause", "resume"} and value:
            vcpu = ws.get(value)
            if vcpu is None:
                return {"error": f"unknown vcpu {value!r}; cores: "
                                 f"{[v.name for v in ws.vcpus()]}"}
            if action == "pause":
                vcpu.pause()
            else:
                vcpu.resume()
            return {"vcpu": value, "status": vcpu.status,
                    "summary": ws.summary_line()}
        if action == "add" and value in {"io", "cpu", "balanced"}:
            vcpu = ws.add_vcpu(kind=value)
            return {"added": vcpu.name, "kind": vcpu.kind,
                    "summary": ws.summary_line()}
        if action == "remove" and value:
            ok = ws.remove_vcpu(value)
            return {"removed": ok, "summary": ws.summary_line()}
        return {"error": f"unsupported action {action!r}"}

    @registry.register(
        "room",
        description=(
            "Project rooms (Prompt 05): persistent per-goal/per-project "
            "workspaces. action=new <title> [--kind goal|project|ad_hoc] "
            "[--linked ID] | list [--status active|paused|archived] | "
            "enter <slug> (mark current, returns ROOM.md) | status <slug> | "
            "archive|pause|resume <slug> | link <a> <b> (cross reference) | "
            "search <query> [--deep] | tick (advance active rooms now) | "
            "stale (idle rooms to review). Work inside a room is sandboxed "
            "to its directory; nothing outside it is touched."
        ),
        capability=Capability.FS_READ,
        parameters={
            "action": ("str — new|list|enter|status|archive|pause|resume|"
                       "link|search|tick|stale"),
            "value": "str (optional) — title for new, slug otherwise",
            "kind": "str (optional) — goal|project|ad_hoc for new",
            "linked": "str (optional) — linked goal/project id for new",
            "status": "str (optional) — status filter for list",
            "deep": "bool (optional) — search files/ contents too",
        },
    )
    def room(*, action: str = "", value: str = "",
             kind: str = "ad_hoc", linked: str = "",
             status: str = "", deep: str = "") -> dict[str, Any]:
        from pathlib import Path as _Path

        from ..workspace.rooms import RoomManager
        root = _Path(context.settings.workspace_dir)
        mgr = RoomManager(root, db=context.db)
        act = (action or "").strip().lower()
        val = (value or "").strip()
        is_deep = str(deep or "").lower() in {"1", "true", "yes", "on"}
        if act == "new" and val:
            r = mgr.create(val, kind=kind or "ad_hoc",
                           linked_id=linked or "")
            return {"room": r.to_dict()}
        if act == "list":
            return {"rooms": [r.to_dict()
                              for r in mgr.list(status=status or "")]}
        if act == "enter" and val:
            with mgr.enter(val):
                pass
            md = (mgr.rooms_dir / val / "ROOM.md").read_text(
                encoding="utf-8")
            return {"slug": val, "room_md": md}
        if act == "status" and val:
            r = mgr.get(val)
            if r is None:
                return {"error": f"no room {val!r}"}
            return {"room": r.to_dict()}
        if act in {"archive", "pause", "resume"} and val:
            r = {"archive": mgr.archive, "pause": mgr.pause,
                 "resume": mgr.resume}[act](val)
            return {"room": r.to_dict()}
        if act == "link" and val:
            parts = val.split()
            if len(parts) != 2:
                return {"error": "link needs two slugs: link <a> <b>"}
            return {"linked": mgr.link(parts[0], parts[1])}
        if act == "search" and val:
            return {"hits": mgr.search(val, deep=is_deep)}
        if act == "tick":
            from ..agents.goals import GoalSystem
            from ..agents.projects import ProjectManager
            return mgr.tick(goal_system=GoalSystem(context),
                            project_manager=ProjectManager(context))
        if act == "stale":
            return {"stale": [r.to_dict() for r in mgr.stale_rooms()]}
        return {"error": f"unsupported room action {act!r}"}
