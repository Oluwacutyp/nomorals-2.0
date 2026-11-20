"""The show — one live run of mocap + rig + cloudcam, through named
states.

The Bevy App pattern, in Python (storyteller-bevy, pattern only,
no upstream code):

* An **App** plugins extend: :meth:`Show.add_plugin` runs a
  plugin's ``build(show)``, which registers systems and resources.
  The three built-in plugins are the organs themselves —
  :class:`MocapPlugin` (the take player), :class:`RigPlugin`
  (drive + render), :class:`CloudCamPlugin` (stage the point
  cloud).
* **States** with ``OnEnter``/``OnExit`` hooks: a show moves
  ``idle → arming → live → paused → live → stopping → done``
  (or ``failed``), and a transition is requested
  (:meth:`Show.request`) and applied at the next schedule
  boundary (:meth:`Show.step`) — never mid-tick, so a state
  change can never cut a phase in half.
* A **Schedule** of ordered phases: every live tick runs
  ``capture → drive → render → publish`` in that order, each
  phase running the systems its plugins registered. A system
  that raises is an event, not a dead show — the tick
  continues and the failure is named in the receipt.
* **Resources**: named, typed, single-owner slots (the store,
  the rig session, the cloudcam session, anything a plugin
  adds). A missing resource is refused by name, never silently
  defaulted.
* **Events**: a bounded journal (drops counted, never silent)
  of everything the show did — state entries/exits, takes
  finishing, held drives, exports.

One show is one run, direct: it holds its own rig session, its
own cloudcam session, its own take player and its own journal.
There is no hub and no shared queue — a second show is a
second, independent run.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..cloudcam.errors import CloudCamError
from ..mocap.errors import MocapError
from ..mocap.kinect_bridge import _ensure_engine
from ..rig.errors import RigError
from .errors import ShowError
from .player import TakePlayer
from .plugins import CloudCamPlugin, MocapPlugin, RigPlugin
from .states import TRANSITIONS, ShowState, legal

__all__ = [
    "PHASES",
    "MAX_EVENTS",
    "MAX_POSED",
    "Show",
]

#: The schedule: the phases of one live tick, in run order.
PHASES: tuple[str, ...] = ("capture", "drive", "render", "publish")

#: Events kept in the journal. Past this, the oldest drops and
#: the drop is counted, never silent.
MAX_EVENTS = 256

#: Posed frames kept for export. Past this, the oldest drops and
#: the drop is counted — an export covers the last 60 seconds at
#: 15 fps, and says so when it had to drop.
MAX_POSED = 900


class Show:
    """One live show: a state machine driving a schedule of
    phases over the mocap store, a rig session and a cloudcam
    session."""

    def __init__(self, name: str, store: Any, *,
                 rig: Any = None, cloudcam: Any = None,
                 fps: float = 15.0,
                 render_dir: str | Path | None = None,
                 export_path: str | Path | None = None) -> None:
        name = (name or "").strip()
        if not name:
            raise ShowError("show name must not be empty",
                            kind="bad_show")
        if store is None:
            raise ShowError(
                f"show {name!r} needs a mocap store — a show with "
                f"nothing to coordinate is not a show",
                kind="bad_show")
        if fps <= 0:
            raise ShowError(f"show fps must be positive, got {fps}",
                            kind="bad_show")
        self.name = name
        self.store = store
        self.fps = float(fps)
        self.render_dir = (Path(render_dir) if render_dir is not None
                           else None)
        self.export_path = (Path(export_path)
                            if export_path is not None else None)
        self.engine = f"show:{name}"
        self.state = ShowState.IDLE
        self._pending: ShowState | None = None
        self._tick = 0
        self._resources: dict[str, Any] = {"store": store}
        if rig is not None:
            self._resources["rig"] = rig
        if cloudcam is not None:
            self._resources["cloudcam"] = cloudcam
        self._systems: dict[str, list[Any]] = {
            phase: [] for phase in PHASES}
        self._plugins: list[Any] = []
        self._enter: dict[ShowState, list[Any]] = {}
        self._exit: dict[ShowState, list[Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._events_dropped = 0
        self.max_events = MAX_EVENTS
        self.max_posed = MAX_POSED
        self._posed: list[Any] = []
        self._posed_dropped = 0
        self._take: TakePlayer | None = None
        self._take_finished_noted = False
        self._last_pose: dict[str, Any] | None = None
        self._last_posed: Any = None
        self._last_frame: bytes | None = None
        self._rendered_bytes = 0
        self._last_cloudcam: dict[str, Any] = {"attached": False}
        self._held = False
        self._was_stale = False
        self._last_skeleton_seq: int | None = None
        self._owns_cloudcam = False
        self._last_error: str | None = None
        self._last_receipt: dict[str, Any] | None = None
        # The built-in plugins: the organs themselves.
        self.add_plugin(MocapPlugin())
        self.add_plugin(RigPlugin())
        self.add_plugin(CloudCamPlugin())
        self.on_enter(ShowState.ARMING, self._arm)
        self.on_enter(ShowState.LIVE, self._go_live)
        self.on_enter(ShowState.STOPPING, self._tear_down)

    # -- identity --------------------------------------------------

    @property
    def pending(self) -> ShowState | None:
        """The requested next state, or None when nothing is
        queued."""
        return self._pending

    @property
    def take(self) -> TakePlayer | None:
        """The loaded take player, or None."""
        return self._take

    @property
    def tick_count(self) -> int:
        """Schedule ticks run so far."""
        return self._tick

    @property
    def last_error(self) -> str | None:
        """The last failure, named, or None."""
        return self._last_error

    @property
    def last_frame(self) -> bytes | None:
        """The last rendered RGB24 frame, or None."""
        return self._last_frame

    @property
    def last_receipt(self) -> dict[str, Any] | None:
        """The last tick's receipt, or None before the first
        tick."""
        return self._last_receipt

    @property
    def posed_count(self) -> int:
        """Posed frames buffered for export."""
        return len(self._posed)

    @property
    def posed_dropped(self) -> int:
        """Posed frames dropped past the buffer bound."""
        return self._posed_dropped

    @property
    def events_dropped(self) -> int:
        """Journal events dropped past the cap."""
        return self._events_dropped

    # -- resources -------------------------------------------------

    def resource(self, name: str) -> Any:
        """A named resource — refused by name when missing."""
        try:
            return self._resources[name]
        except KeyError as exc:
            raise ShowError(
                f"show {self.name!r} has no resource {name!r}",
                kind="missing_resource") from exc

    def add_resource(self, name: str, value: Any) -> None:
        """Register a resource. A duplicate name is refused —
        one resource per name, like one Bevy resource per type."""
        if name in self._resources:
            raise ShowError(
                f"show {self.name!r} already has resource {name!r}",
                kind="duplicate_resource")
        self._resources[name] = value

    def set_resource(self, name: str, value: Any) -> None:
        """Replace (or add) a resource."""
        self._resources[name] = value

    # -- plugins + systems ------------------------------------------

    def add_plugin(self, plugin: Any) -> None:
        """Run a plugin's ``build(show)``. A plugin that needs a
        missing resource fails here, by name — at build time,
        never mid-show."""
        plugin.build(self)
        self._plugins.append(plugin)

    def system(self, phase: str, fn: Any) -> None:
        """Register a system into a phase of the schedule. An
        unknown phase is refused by name."""
        if phase not in PHASES:
            raise ShowError(
                f"show {self.name!r} has no phase {phase!r} — "
                f"phases are {', '.join(PHASES)}",
                kind="bad_phase")
        self._systems[phase].append(fn)

    def on_enter(self, state: ShowState, fn: Any) -> None:
        """Run ``fn(show)`` when the show enters ``state``."""
        self._enter.setdefault(state, []).append(fn)

    def on_exit(self, state: ShowState, fn: Any) -> None:
        """Run ``fn(show)`` when the show leaves ``state``."""
        self._exit.setdefault(state, []).append(fn)

    # -- takes ------------------------------------------------------

    def load_take(self, path: str | Path) -> dict[str, Any]:
        """Load a take from a BVH (``.bvh``) or JSON (``.json``)
        file. Anything else is refused by name."""
        target = Path(path).expanduser()
        suffix = target.suffix.lower()
        if suffix == ".bvh":
            player = TakePlayer.from_bvh(target)
        elif suffix == ".json":
            player = TakePlayer.from_json(target)
        else:
            raise ShowError(
                f"show take {target} must be .bvh or .json, got "
                f"{suffix or 'no suffix'}",
                kind="bad_take")
        self._take = player
        self._take_finished_noted = False
        return player.to_dict()

    def load_take_frames(self, skeleton: Any,
                         frames: list[Any]) -> dict[str, Any]:
        """Load a take from an in-memory skeleton + frames (the
        embedding and test path)."""
        self._take = TakePlayer(skeleton, frames)
        self._take_finished_noted = False
        return self._take.to_dict()

    def attach_rig(self, session: Any) -> dict[str, Any]:
        """Attach a rig session (replacing any attached one)."""
        if session is None:
            raise ShowError("attach_rig needs a rig session",
                            kind="bad_rig")
        self.set_resource("rig", session)
        return {"ok": True, "show": self.name,
                "rig": session.rig.name}

    def attach_cloudcam(self, session: Any) -> dict[str, Any]:
        """Attach a cloudcam session (replacing any attached one).
        A show-owned poller that is still running is stopped
        first — replacing a live stream must not leak its
        thread."""
        if session is None:
            raise ShowError("attach_cloudcam needs a cloudcam "
                            "session", kind="bad_cloudcam")
        old = self._resources.get("cloudcam")
        if (old is not None and old is not session
                and self._owns_cloudcam and old.running):
            try:
                old.stop()
            except Exception as exc:  # noqa: BLE001 - best effort
                self._record("cloudcam_stop_failed",
                             {"error": str(exc)})
        self._owns_cloudcam = False
        self.set_resource("cloudcam", session)
        return {"ok": True, "show": self.name,
                "cloudcam": session.name}

    # -- the state machine ------------------------------------------

    def request(self, to_state: ShowState) -> ShowState:
        """Queue the next state. An illegal transition is refused
        now, naming both states and the legal exits — but the
        transition itself still applies only at the next
        schedule boundary, in :meth:`step`."""
        if not legal(self.state, to_state):
            exits = sorted(s.name for s in TRANSITIONS[self.state])
            raise ShowError(
                f"show {self.name!r} cannot go from "
                f"{self.state.name} to {to_state.name} — from "
                f"{self.state.name} it can go to "
                f"{', '.join(exits) or 'nowhere'}",
                kind="bad_transition")
        self._pending = to_state
        return to_state

    def step(self) -> dict[str, Any]:
        """One schedule tick: apply a pending transition, then —
        when live — run the phases in order. Returns the tick's
        receipt."""
        if self._pending is not None:
            to_state = self._pending
            self._pending = None
            self._apply(to_state)
        self._tick += 1
        receipt: dict[str, Any] = {
            "show": self.name,
            "state": self.state.value,
            "tick": self._tick,
            "phases": {},
        }
        if self.state is ShowState.LIVE:
            for phase in PHASES:
                errors: list[str] = []
                ran = 0
                for fn in self._systems[phase]:
                    try:
                        fn(self)
                        ran += 1
                    except Exception as exc:  # noqa: BLE001
                        kind = getattr(exc, "kind", "system_error")
                        errors.append(f"{kind}: {exc}")
                        self._record("system_error", {
                            "phase": phase, "kind": kind,
                            "error": str(exc)})
                receipt["phases"][phase] = {"systems": ran,
                                            "errors": errors}
        else:
            receipt["phases"] = {
                phase: {"systems": 0, "errors": []}
                for phase in PHASES}
        receipt["pose"] = self._last_pose
        receipt["held"] = self._held
        receipt["rendered_bytes"] = self._rendered_bytes
        receipt["cloudcam"] = dict(self._last_cloudcam)
        self._last_receipt = receipt
        return receipt

    def _apply(self, to_state: ShowState) -> None:
        """Run a queued transition: exit hooks, the state change,
        enter hooks. A hook that fails by name leaves the show
        FAILED with the reason recorded — and the failure is
        raised to the caller too, so a refused start is never
        silent."""
        from_state = self.state
        try:
            for fn in self._exit.get(from_state, []):
                fn(self)
            self.state = to_state
            for fn in self._enter.get(to_state, []):
                fn(self)
        except (ShowError, MocapError, RigError,
                CloudCamError) as exc:
            kind = getattr(exc, "kind", "show_error")
            self._fail(f"{kind}: {exc}")
            raise
        self._record("state_exit", {"from": from_state.value,
                                    "to": to_state.value})
        self._record("state_enter", {"from": from_state.value,
                                     "to": to_state.value})

    def _fail(self, reason: str) -> None:
        """Mark the show FAILED with the reason, and tear down
        what arming claimed — a failed arm must never leak a
        channel claim or a poller thread."""
        self.state = ShowState.FAILED
        self._last_error = reason
        self._record("show_failed", {"reason": reason})
        self._tear_down_quietly()

    # -- the lifecycle verbs ----------------------------------------

    def start(self) -> dict[str, Any]:
        """Arm, then go live. Arming claims the skeleton channel
        and starts the attached cloudcam; going live requires a
        rig, by name."""
        self.request(ShowState.ARMING)
        self.step()
        self.request(ShowState.LIVE)
        self.step()
        return self.status()

    def pause(self) -> dict[str, Any]:
        """Hold the show: phases stop, the pose holds, the
        cloudcam poller keeps running into its ring."""
        self.request(ShowState.PAUSED)
        self.step()
        return self.status()

    def resume(self) -> dict[str, Any]:
        """Resume a paused show."""
        self.request(ShowState.LIVE)
        self.step()
        return self.status()

    def stop(self) -> dict[str, Any]:
        """Stop: export when configured, release the skeleton
        claim, stop the show-owned cloudcam, then done."""
        self.request(ShowState.STOPPING)
        self.step()
        self.request(ShowState.DONE)
        self.step()
        return self.status()

    def run(self, ticks: int, interval_s: float = 0.0) -> dict[str, Any]:
        """Run the show for a bounded number of ticks. Starts it
        when idle/done/failed, resumes it when paused. A tick is
        one schedule step; ``interval_s`` paces them. Stops early
        when the show leaves LIVE."""
        if not isinstance(ticks, int) or ticks < 1 or ticks > 3600:
            raise ShowError(
                f"show run needs ticks 1..3600, got {ticks}",
                kind="bad_run")
        if interval_s < 0:
            raise ShowError(
                f"show run interval must not be negative, got "
                f"{interval_s}", kind="bad_run")
        if self.state in (ShowState.IDLE, ShowState.DONE,
                          ShowState.FAILED):
            self.start()
        elif self.state is ShowState.PAUSED:
            self.resume()
        for _ in range(ticks):
            self.step()
            if self.state is not ShowState.LIVE:
                break
            if interval_s:
                time.sleep(interval_s)
        return self.status()

    def export(self, path: str | Path) -> dict[str, Any]:
        """Write the buffered posed frames out as BVH (the rig
        skeleton + driven rotations) — the format game engines
        read. The export is recorded on the store's ``bvh``
        channel."""
        rig = self.resource("rig")
        if not self._posed:
            raise ShowError(
                f"show {self.name!r} has driven no poses yet — "
                f"start it and tick first",
                kind="no_poses")
        return rig.export_bvh(list(self._posed), path,
                              store=self.store)

    # -- hooks ------------------------------------------------------

    def _arm(self, _show: Show) -> None:
        """Entering ARMING: register the show's engine, reset the
        take, claim the skeleton channel when a take is loaded,
        and start the attached cloudcam when it is not running."""
        _ensure_engine(
            self.store, self.engine,
            produces=("skeleton",) if self._take is not None else (),
            note=f"live show {self.name!r} — take player and "
                 f"drive")
        if self._take is not None:
            self._take.reset()
            self._take_finished_noted = False
            self.store.claim("skeleton", self.engine)
        cam = self._resources.get("cloudcam")
        if cam is not None and not cam.running:
            cam.start()
            self._owns_cloudcam = True

    def _go_live(self, _show: Show) -> None:
        """Entering LIVE: a show with no rig has nothing to
        drive — refused by name."""
        if "rig" not in self._resources:
            raise ShowError(
                f"show {self.name!r} has no rig — attach one "
                f"(show attach-rig) before going live",
                kind="no_rig")

    def _tear_down(self, _show: Show) -> None:
        """Entering STOPPING: export when configured, release the
        skeleton claim, stop the show-owned cloudcam, drop the
        engine record. Every step is guarded — a stop must
        complete even when one piece is already gone."""
        if self.export_path is not None and self._posed:
            try:
                result = self.export(self.export_path)
                self._record("export_done", result)
            except Exception as exc:  # noqa: BLE001 - stop completes
                self._last_error = str(exc)
                self._record("export_failed", {"error": str(exc)})
        self._tear_down_quietly()

    def _tear_down_quietly(self) -> None:
        """Release what arming claimed, best effort, never
        raising — the quiet half of teardown, shared by the
        STOPPING hook and the failure path."""
        try:
            if self.store.owner("skeleton") == self.engine:
                self.store.release("skeleton", self.engine)
        except Exception as exc:  # noqa: BLE001 - best effort
            self._record("release_failed", {"error": str(exc)})
        cam = self._resources.get("cloudcam")
        if cam is not None and self._owns_cloudcam and cam.running:
            try:
                cam.stop()
            except Exception as exc:  # noqa: BLE001 - best effort
                self._record("cloudcam_stop_failed",
                             {"error": str(exc)})
        self._owns_cloudcam = False
        try:
            names = {e["name"] for e in self.store.engines()}
            if self.engine in names:
                self.store.unregister_engine(self.engine)
        except Exception as exc:  # noqa: BLE001 - best effort
            self._record("unregister_failed", {"error": str(exc)})

    # -- events + status --------------------------------------------

    def _record(self, kind: str, detail: dict[str, Any]) -> None:
        """Append an event to the bounded journal."""
        self._events.append({
            "kind": kind,
            "tick": self._tick,
            "at": time.monotonic(),
            "detail": dict(detail),
        })
        while len(self._events) > MAX_EVENTS:
            self._events.pop(0)
            self._events_dropped += 1

    def events(self, limit: int | None = None) -> list[dict[str, Any]]:
        """The journal, oldest first — the last ``limit`` events
        when given, all of them otherwise."""
        if limit is None:
            return [dict(e) for e in self._events]
        return [dict(e) for e in self._events[-limit:]]

    def status(self) -> dict[str, Any]:
        """Everything a status line needs, honest numbers only."""
        rig = self._resources.get("rig")
        cam = self._resources.get("cloudcam")
        return {
            "name": self.name,
            "state": self.state.value,
            "tick": self._tick,
            "fps": self.fps,
            "resources": sorted(self._resources),
            "take": (self._take.to_dict()
                     if self._take is not None else None),
            "rig": (rig.rig.name if rig is not None else None),
            "cloudcam": (cam.status() if cam is not None else None),
            "posed": len(self._posed),
            "posed_dropped": self._posed_dropped,
            "events": len(self._events),
            "events_dropped": self._events_dropped,
            "last_error": self._last_error,
            "last_receipt": self._last_receipt,
        }
