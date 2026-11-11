"""The computer organ's facade: profile gating + one call surface.

Everything above the organ (the ``computer`` tool, the /computer
chat commands, missions) talks to ComputerService. It owns the
profile gate — the full computer exists in code everywhere, but a
profile with no display facility (Termux/phone, constrained
embedded) reports itself unavailable WITH the reason rather than
pretending — and it wires the brain-health probe into the session
gate: when a runtime context is attached, the gate reads
llm.power's state and only a *full* brain may act; degraded or
offline means observe-only, per the personal-surface law.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from typing import Any

from ..core import profile as _profile_mod
from . import failures, loop, provision
from .protocol import DriverClient
from .sessions import GateRefused, SessionError, SessionManager, SessionRecord


class ComputerService:
    """Facade over the session manager, with gating and wiring."""

    def __init__(self, context: Any = None,
                 manager: SessionManager | None = None) -> None:
        self.context = context
        if manager is None:
            probe: Callable[[], bool] | None = None
            if context is not None:
                probe = self._brain_ok
            manager = SessionManager(brain_probe=probe)
        self.manager = manager
        self._availability_cache: tuple[float, dict[str, Any]] | None = None

    # -- wiring -----------------------------------------------------------

    def _brain_ok(self) -> bool:
        """True only when the attached brain is fully healthy."""
        try:
            from ..llm.power import power_state
            return power_state(self.context) == "full"
        except Exception as exc:  # noqa: BLE001 — unwired brain is not healthy
            failures.record_failure("service brain probe", exc)
            return False

    def _complete_fn(self) -> Callable[[str], str] | None:
        """A model completion callable from the attached context."""
        router = getattr(self.context, "router", None) \
            if self.context is not None else None
        complete = getattr(router, "complete", None)
        if complete is None:
            return None

        def _fn(prompt: str) -> str:
            response = complete(prompt, task_kind="computer")
            text = getattr(response, "text", None)
            return str(text if text is not None else response)

        return _fn

    def _vision_fn(self) -> Callable[[bytes, str], str] | None:
        """A vision callable from the attached context's router.

        The desktop decider's eyes: the router's ``describe_image``
        routes the perception's real PNG to a VL-capable model.
        None when no router/vision rung is attached — the decider
        then answers from the text summary, as before.
        """
        router = getattr(self.context, "router", None) \
            if self.context is not None else None
        describe = getattr(router, "describe_image", None)
        if describe is None:
            return None

        def _fn(png: bytes, prompt: str) -> str:
            response = describe(png, prompt)
            if getattr(response, "ok", True) is False:
                raise RuntimeError(
                    f"vision model declined: "
                    f"{getattr(response, 'error', '') or 'no answer'}")
            text = getattr(response, "text", None)
            return str(text if text is not None else response)

        return _fn

    # -- availability / profile gating -------------------------------------

    def availability(self) -> dict[str, Any]:
        """Can this host/profile serve a desktop right now, and why."""
        now = time.time()
        if self._availability_cache and \
                now - self._availability_cache[0] < 30.0:
            return dict(self._availability_cache[1])
        result = self._probe_availability()
        self._availability_cache = (now, result)
        return dict(result)

    def _probe_availability(self) -> dict[str, Any]:
        profile = _profile_mod.detect_profile()
        base: dict[str, Any] = {
            "profile": profile.kind,
            "platform": sys.platform,
            "available": False,
            "reason": "",
            "backend": "",
        }
        if profile.kind == "termux":
            base["reason"] = (
                "this device runs a Termux/Android profile — there is "
                "no desktop display facility to drive here. The "
                "computer lives on hosts with a desktop (her server, "
                "a workstation); this profile reports it unavailable "
                "rather than pretending.")
            return base
        if profile.kind == "embedded":
            base["reason"] = (
                "this is a constrained embedded profile — no display "
                "facility and no room to virtualise one.")
            return base
        try:
            bridge = provision.ensure_bridge()
        except provision.ProvisionError as exc:
            base["reason"] = str(exc)
            return base
        base["backend"] = "desktop bridge (" + bridge.how + ")"
        if sys.platform.startswith("win"):
            info = self._driver_probe(bridge_path=bridge.path, env=None)
            base["available"] = bool(info.get("available"))
            base["reason"] = str(info.get("reason") or
                                 "the console desktop answers")
            return base
        display = os.environ.get("NM_COMPUTER_DISPLAY") or \
            os.environ.get("DISPLAY")
        if display:
            env = dict(os.environ)
            env["DISPLAY"] = display
            info = self._driver_probe(bridge_path=bridge.path, env=env)
            base["available"] = bool(info.get("available"))
            base["reason"] = str(
                info.get("reason")
                or f"the host display {display} answers")
            return base
        # No display in the environment: a virtual one is the route.
        if provision.find_xvfb():
            base["available"] = True
            base["reason"] = ("no host display; Devon will start her "
                              "own virtual display (Xvfb) per session")
            return base
        try:
            provision.ensure_xvfb_binary()
            base["available"] = True
            base["reason"] = ("no host display; Devon provisioned a "
                              "virtual display server (Xvfb)")
            return base
        except provision.ProvisionError as exc:
            base["reason"] = str(exc)
            return base

    @staticmethod
    def _driver_probe(bridge_path: Any,
                      env: dict[str, str] | None) -> dict[str, Any]:
        client = DriverClient([str(bridge_path)], env=env,
                              name="desktop-bridge[probe]")
        try:
            client.start()
            return client.probe()
        except Exception as exc:  # noqa: BLE001 — probe reports, not raises
            return {"available": False, "reason": str(exc)}
        finally:
            client.close()

    # -- surfaces -----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        avail = self.availability()
        sessions = []
        for rec in self.manager.sessions():
            row = rec.describe()
            row["gate"] = self.manager.gate_reasons(rec)
            sessions.append(row)
        return {
            "available": avail["available"],
            "reason": avail["reason"],
            "profile": avail["profile"],
            "platform": avail["platform"],
            "backend": avail["backend"],
            "sessions": sessions,
            "failures": failures.failure_count(),
        }

    def status_text(self) -> str:
        st = self.status()
        lines = [
            "computer: " + ("available" if st["available"]
                            else "UNAVAILABLE"),
            f"  {st['reason']}",
            f"  profile {st['profile']} · platform {st['platform']}"
            + (f" · backend {st['backend']}" if st["backend"] else ""),
        ]
        if not st["sessions"]:
            lines.append("  no desktop sessions yet — /computer open "
                         "starts one.")
        for row in st["sessions"]:
            gate = "; ".join(row["gate"]) if row["gate"] else "hands free"
            lines.append(
                f"  • {row['id']} [{row['state']}] {row['mode']} "
                f"display {row['display']} {row['width']}x{row['height']}"
                f" — {gate}")
            if row["note"]:
                lines.append(f"      {row['note']}")
        if st["failures"]:
            lines.append(f"  recorded failures: {st['failures']} "
                         "(latest in the computer ledger)")
        return "\n".join(lines)

    # -- session verbs -------------------------------------------------------

    def _resolve(self, session_id: str = "") -> SessionRecord:
        rec = self.manager.resolve(session_id)
        if rec is None:
            raise SessionError(
                "no desktop session exists yet — open one first")
        return rec

    def open(self, mission: str = "") -> dict[str, Any]:
        avail = self.availability()
        if not avail["available"]:
            return {"ok": False, "error": avail["reason"]}
        try:
            rec = self.manager.open_session(mission=mission)
        except SessionError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "session": rec.describe()}

    def close(self, session_id: str = "") -> dict[str, Any]:
        try:
            rec = self._resolve(session_id)
            closed = self.manager.close_session(rec.id)
        except SessionError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "session": closed.describe()}

    def perceive(self, session_id: str = "") -> dict[str, Any]:
        """Watch: one perception (always allowed on an open session)."""
        try:
            rec = self._resolve(session_id)
            seen = loop.capture(self.manager, rec)
        except SessionError as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "session": rec.id,
            "width": seen.width,
            "height": seen.height,
            "frame_path": seen.frame_path,
            "frame_hash": seen.frame_hash,
            "windows": seen.windows,
            "focused_id": seen.focused_id,
            "pointer": seen.pointer,
            "summary": seen.summary(),
        }

    def act(self, kind: str, args: dict[str, Any] | None = None,
            session_id: str = "") -> dict[str, Any]:
        """One gated action by kind (click/type/key/launch/...)."""
        try:
            rec = self._resolve(session_id)
            action = loop.Action(kind=kind, args=args or {})
            if kind == "wait":
                return {"ok": False,
                        "error": "wait is a loop step, not a single act"}
            method, params = loop.action_to_call(action)
            result = self.manager.act(rec, method, params)
        except GateRefused as refused:
            return {"ok": False, "gate": refused.reasons,
                    "error": str(refused)}
        except (SessionError, KeyError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "result": result}

    def run(self, goal: str, session_id: str = "",
            max_steps: int = 25,
            complete_fn: Callable[[str], str] | None = None,
            vision_fn: Callable[[bytes, str], str] | None = None,
            ) -> dict[str, Any]:
        """A brain-directed run toward ``goal`` on a session."""
        fn = complete_fn or self._complete_fn()
        if fn is None:
            return {"ok": False,
                    "error": "no model is wired in this context, so "
                             "there is no brain to direct the desktop "
                             "— perceive/act still work, and runs need "
                             "the tool or chat surface's model."}
        eyes = vision_fn if vision_fn is not None else self._vision_fn()
        try:
            rec = self._resolve(session_id)
            report = loop.run_task(
                self.manager, rec, goal,
                loop.LLMDecider(fn, vision_fn=eyes),
                max_steps=max_steps)
        except SessionError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": report.status == "completed",
                "status": report.status, "reason": report.reason,
                "report": report.describe()}

    # -- owner controls -------------------------------------------------------

    def takeover(self, session_id: str = "",
                 by: str = "owner") -> dict[str, Any]:
        return self._flag(session_id,
                          lambda rec: self.manager.set_takeover(
                              rec.id, True, by=by))

    def handback(self, session_id: str = "") -> dict[str, Any]:
        return self._flag(session_id,
                          lambda rec: self.manager.set_takeover(
                              rec.id, False))

    def stop(self, session_id: str = "") -> dict[str, Any]:
        return self._flag(session_id,
                          lambda rec: self.manager.set_owner_stop(
                              rec.id, True))

    def resume(self, session_id: str = "") -> dict[str, Any]:
        return self._flag(session_id,
                          lambda rec: self.manager.set_owner_stop(
                              rec.id, False))

    def _flag(self, session_id: str,
              apply: Callable[[SessionRecord], SessionRecord],
              ) -> dict[str, Any]:
        try:
            rec = self._resolve(session_id)
            updated = apply(rec)
        except SessionError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "session": updated.describe(),
                "gate": self.manager.gate_reasons(updated)}


def default_service(context: Any = None) -> ComputerService:
    """The service the tool and chat surfaces share."""
    return ComputerService(context=context)
