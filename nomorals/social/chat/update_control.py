"""Chat-facing ``/update`` / ``/pull`` / ``/deploy`` / ``/restart``.

These commands already existed as ``nm update`` on the console
(:mod:`nomorals.os.update` — transactional pull + migrate + health-check
+ rollback) but were NOT reachable from Telegram: the control registry
had no such kinds, so the owner could not pull the bot's files from chat.
This module is the chat layer on top of that engine — it adds the one
thing the console path never needed: the **mission gate**.

Flow for every kind (drain → checkpoint → restart → resume):

1. **Gate.** List non-terminal missions from the
   :class:`~nomorals.missions.mission.MissionStore`. Any live mission →
   refuse, naming the missions. No update ever runs mid-mission.
2. **Drain + checkpoint.** ``/update``/``/deploy``/``/restart`` checkpoint
   every resumable mission via ``MissionStore.checkpoint`` (append-only,
   immutable) so a restart resumes from the checkpoint instead of losing
   work. ``/pull`` only pulls code — it checkpoints nothing and restarts
   nothing, because the running process keeps its already-loaded code.
3. **Apply.** ``/pull``/``/update``/``/deploy`` delegate the code move to
   the ``update_runner`` (default: ``nomorals.os.update.UpdateManager``);
   ``/restart`` skips the pull.
4. **Restart + resume.** ``/update``/``/deploy``/``/restart`` return
   ``restart_requested=True``; the runtime performs the actual process
   restart (stop + exec) and, on boot, ``PartnerRuntime`` resumes the
   interrupted missions (dead/stale driver) on a background thread;
   :func:`resume_after_restart` reports the resumable inventory here.

The controller never restarts the process itself and never touches git
directly — both are injected callables, so tests prove the gate and the
ordering without ever pulling or restarting anything.

Owner-only: these kinds are absent from ``PUBLIC_COMMANDS``, so
:func:`caller_may_run` denies non-owners in the policy layer — the menu
never lists them for non-owners, but hiding is not the enforcement.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "UPDATE_KINDS",
    "ChatUpdateController",
    "resume_after_restart",
    "active_missions",
    "checkpoint_all",
]

#: The four chat update commands, all owner-only (not in PUBLIC_COMMANDS).
UPDATE_KINDS: tuple[str, ...] = ("update", "pull", "deploy", "restart")


def active_missions(mission_store: Any) -> list[Any]:
    """Non-terminal missions from a MissionStore (or duck-typed store)."""
    if mission_store is None:
        return []
    try:
        missions = mission_store.list(active_only=True, limit=200)
    except TypeError:
        missions = mission_store.list()
    out = []
    for m in missions or []:
        terminal = False
        if hasattr(m, "terminal"):
            terminal = bool(m.terminal)
        elif isinstance(m, dict):
            terminal = str(m.get("status", "")).lower() in (
                "done", "completed", "failed", "cancelled", "archived")
        if not terminal:
            out.append(m)
    return out


def _mission_label(m: Any) -> str:
    name = getattr(m, "name", "") or (m.get("name") if isinstance(m, dict) else "")
    mid = getattr(m, "id", "") or (m.get("id") if isinstance(m, dict) else "")
    goal = getattr(m, "goal", "") or (m.get("goal") if isinstance(m, dict) else "")
    title = name or (str(goal)[:60] if goal else str(mid))
    return f"{title} [{str(mid)[:8]}]" if mid else title


def checkpoint_all(mission_store: Any, *, label: str = "pre-restart") -> list[str]:
    """Checkpoint every resumable mission. Returns checkpoint ids."""
    if mission_store is None:
        return []
    ids: list[str] = []
    try:
        resumable = mission_store.resumable()
    except Exception:  # noqa: BLE001 - resumable() is best-effort
        resumable = []
    for mission in resumable or []:
        try:
            cp = mission_store.checkpoint(mission, label=label)
            ids.append(str(getattr(cp, "id", cp)))
        except Exception:  # noqa: BLE001 - one bad mission must not block the rest
            continue
    return ids


def resume_after_restart(mission_store: Any) -> str:
    """Report the resumable inventory after a restart.

    On boot, ``PartnerRuntime``'s boot-resumption sweep drives
    interrupted missions — RUNNING (or PENDING with drive evidence)
    whose driver is dead or stale — back to completion on a
    background thread; PAUSED missions stay paused. This function is
    only the chat-visible inventory of what is resumable — never a
    claim that resumption already happened.
    """
    if mission_store is None:
        return "no mission store attached — nothing to resume."
    try:
        resumable = mission_store.resumable()
    except Exception as exc:  # noqa: BLE001
        return f"could not list resumable missions: {exc}"
    if not resumable:
        return "no resumable missions — nothing to resume after restart."
    lines = ["resumable after restart:"]
    for m in resumable:
        lines.append(f"  • {_mission_label(m)}")
    return "\n".join(lines)


class ChatUpdateController:
    """Gate + checkpoint + delegate for the four chat update commands."""

    def __init__(self, *, mission_store: Any = None,
                 update_runner: Callable[[bool], Any] | None = None) -> None:
        #: MissionStore (or duck-typed); None → fail-closed (refuse).
        self.mission_store = mission_store
        #: ``update_runner(pull: bool) -> report`` — default None means
        #: "no update engine attached" (refuse pull/update/deploy, but
        #: restart still works: gate + checkpoint + restart).
        self.update_runner = update_runner

    # -- gate ------------------------------------------------------------
    def gate(self) -> tuple[bool, list[Any], str]:
        """``(ok, active, message)`` — ok iff no live mission and a store
        is attached. Fail-closed: no store → refuse, never proceed."""
        if self.mission_store is None:
            return (False, [],
                    "refusing: no mission store attached — cannot verify "
                    "no mission is running. Wire the MissionStore before "
                    "using /update from chat.")
        live = active_missions(self.mission_store)
        if live:
            names = ", ".join(_mission_label(m) for m in live)
            return (False, live,
                    f"refusing: {len(live)} mission(s) still running — "
                    f"{names}. Wait for them (or /mission cancel) and retry. "
                    "Updates never run mid-mission.")
        return (True, [], "gate clear: no live missions.")

    # -- execute ---------------------------------------------------------
    def execute(self, kind: str) -> dict[str, Any]:
        """Run one update command. Never raises; returns a report dict::

            {"ok", "message", "restart_requested", "checkpoints",
             "update_report"}

        ``restart_requested`` tells the runtime to perform the actual
        process restart (stop + exec) after replying — the controller
        never restarts the process itself.
        """
        kind = (kind or "").strip().lower()
        if kind not in UPDATE_KINDS:
            return {"ok": False, "message": f"unknown update command /{kind}",
                    "restart_requested": False, "checkpoints": [],
                    "update_report": None}
        ok, _live, gate_msg = self.gate()
        if not ok:
            return {"ok": False, "message": gate_msg,
                    "restart_requested": False, "checkpoints": [],
                    "update_report": None}
        if kind == "pull":
            # Pull only: gate passed, no checkpoint/restart — the running
            # process keeps its loaded code; new code takes effect on the
            # next restart.
            return self._apply_pull(restart=False)
        if kind == "restart":
            checkpoints = checkpoint_all(self.mission_store, label="pre-restart")
            return {"ok": True,
                    "message": ("restarting: " + gate_msg + " "
                                f"{len(checkpoints)} mission checkpoint(s) taken. "
                                + resume_after_restart(self.mission_store)),
                    "restart_requested": True, "checkpoints": checkpoints,
                    "update_report": None}
        # update / deploy: checkpoint, pull, restart.
        checkpoints = checkpoint_all(self.mission_store, label=f"pre-{kind}")
        result = self._apply_pull(restart=True)
        result["checkpoints"] = checkpoints
        return result

    def _apply_pull(self, *, restart: bool) -> dict[str, Any]:
        if self.update_runner is None:
            return {"ok": False,
                    "message": "refusing: no update engine attached — "
                               "wire nomorals.os.update.UpdateManager before "
                               "using /pull /update /deploy from chat.",
                    "restart_requested": False, "checkpoints": [],
                    "update_report": None}
        try:
            report = self.update_runner(True)
        except Exception as exc:  # noqa: BLE001 - runner failure is a report, not a crash
            return {"ok": False, "message": f"update engine failed: {exc}",
                    "restart_requested": False, "checkpoints": [],
                    "update_report": None}
        ok = bool(getattr(report, "ok", False)) if not isinstance(report, dict) else bool(report.get("ok"))
        render = ""
        if hasattr(report, "render"):
            try:
                render = report.render()
            except Exception:  # noqa: BLE001
                render = str(report)
        else:
            render = str(report)
        if not ok:
            return {"ok": False,
                    "message": f"update failed — NOT restarting:\n{render}",
                    "restart_requested": False, "checkpoints": [],
                    "update_report": report}
        suffix = " restarting now." if restart else " new code takes effect on the next restart."
        return {"ok": True,
                "message": f"update OK —{suffix}\n{render}",
                "restart_requested": restart, "checkpoints": [],
                "update_report": report}
