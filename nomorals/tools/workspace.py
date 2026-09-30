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
