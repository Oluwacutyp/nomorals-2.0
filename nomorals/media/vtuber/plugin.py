"""The VTuber plugin — a performance, as a show plugin.

A show plugin registers systems into the show's schedule; the
VTuber plugin registers three, after the built-ins (so they run
after the take players and the rig driver, in each phase):

* ``capture`` — step the act (fire due beats) and write the act's
  face overlay into the face channel: the active expression's
  blendshapes merged over the face take's current frame (the act
  wins on the keys it names), plus the active line as text, so the
  rig's text→viseme path speaks it.
* ``drive`` — merge the active overlays into the driven pose
  (expression blendshapes + gesture rotations, the act wins on
  what it names) and replace the buffered pose too, so the BVH
  export covers the performance, not the raw take.
* ``render`` — apply the active camera shot to the last rendered
  frame (a crop; the PNGs the rig plugin wrote stay full-frame
  and the receipt says so).

A performance is per-show: ``attach_performance`` binds one act
to one show (a second attach is refused — detach first), and
``detach_performance`` stops it. Live capture is separate:
``attach_live`` binds a live MVN receiver (already writing into
the show's store) by name and holds the recorded take while it is
attached — a take and a live stream would fight over the same
skeleton channel, and the rule is the live stream wins.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..mocap.errors import MocapError
from ..show.errors import ShowError
from ..show.plugins import ShowPlugin
from .act import Act
from .camera import apply_shot
from .character import Character
from .errors import VtuberError
from .player import ActPlayer

__all__ = [
    "VtuberPlugin",
    "attach_live",
    "attach_performance",
    "detach_live",
    "detach_performance",
]


class VtuberPlugin(ShowPlugin):
    """The performance, as systems on a show."""

    def __init__(self, player: ActPlayer) -> None:
        self.player = player

    def build(self, show: Any) -> None:
        show.system("capture", self._capture)
        show.system("drive", self._drive)
        show.system("render", self._render)

    # -- systems -----------------------------------------------------

    def _capture(self, show: Any) -> None:
        player = self.player
        if not player._attached:
            return
        player.tick(show)
        blend, text = player.active_face()
        if not blend and text is None:
            return
        base_blend: dict[str, float] = {}
        base_bones: dict[str, Any] = {}
        try:
            view = show.store.read("face")
            payload = view.get("payload") or {}
            if isinstance(payload.get("blendshapes"), dict):
                base_blend = dict(payload["blendshapes"])
            if isinstance(payload.get("bones"), dict):
                base_bones = dict(payload["bones"])
        except MocapError:
            pass
        merged = {**base_blend, **blend}
        frame: dict[str, Any] = {
            "blendshapes": merged,
            "bones": base_bones,
            "frame": show.tick_count,
            "time_s": player.now_s,
        }
        if text is not None:
            frame["text"] = text
        show.store.write("face", show.engine, frame,
                         kind="snapshot")
        show._last_face = {"attached": True, "vtuber": True,
                           "saying": text}

    def _drive(self, show: Any) -> None:
        player = self.player
        if not player._attached:
            return
        pose = show._last_posed
        if pose is None:
            return
        blend, rotations = player.active_pose()
        if not blend and not rotations:
            return
        merged = replace(
            pose,
            blendshapes={**pose.blendshapes, **blend},
            rotations={**pose.rotations, **rotations})
        show._last_posed = merged
        show._last_pose = merged.to_dict()
        if show._posed and show._posed[-1] is pose:
            show._posed[-1] = merged

    def _render(self, show: Any) -> None:
        player = self.player
        if not player._attached:
            return
        frame = show._last_frame
        if frame is None:
            return
        shot = player.current_shot
        show._last_shot = shot
        if shot == "wide":
            return
        rig = show.resource("rig")
        rgb, width, height = apply_shot(
            frame, rig.width, rig.height, shot)
        show._last_frame = rgb
        show._last_frame_size = (width, height)
        show._record("vtuber_shot", {
            "shot": shot, "width": width, "height": height})


# -- attach / detach -------------------------------------------------

def attach_performance(show: Any, act: Act,
                       character: Character) -> ActPlayer:
    """Bind an act to a show and start performing it. A second
    performance on the same show is refused — detach first."""
    if getattr(show, "_vtuber_player", None) is not None:
        raise VtuberError(
            f"show {show.name!r} already performs "
            f"{show._vtuber_player.act.name!r} — detach it first "
            f"(vtuber stop)",
            kind="already_performing")
    try:
        rig_session = show.resource("rig")
    except ShowError as exc:
        raise VtuberError(
            f"show {show.name!r} has no rig — attach one before "
            f"performing (show attach-rig)",
            kind="no_rig") from exc
    player = ActPlayer(act, character, rig_session.rig)
    show._vtuber_player = player
    show._last_shot = "wide"
    show.add_plugin(VtuberPlugin(player))
    show._record("vtuber_attached", {
        "act": act.name, "character": character.name,
        "beats": len(act.beats)})
    return player


def detach_performance(show: Any) -> dict[str, Any]:
    """Stop the performance on a show."""
    player = getattr(show, "_vtuber_player", None)
    if player is None:
        raise VtuberError(
            f"show {show.name!r} performs nothing — nothing to "
            f"detach",
            kind="not_performing")
    player.detach()
    show._vtuber_player = None
    show._record("vtuber_detached", {"act": player.act.name})
    return {"ok": True, "show": show.name, "act": player.act.name}


def attach_live(show: Any, receiver: Any, name: str) -> dict[str, Any]:
    """Bind a live MVN receiver to a show, by name. The receiver
    must already write into the show's store — a receiver on
    another store would drive nothing and lie about it. While
    attached, the recorded take is held: one skeleton channel,
    one writer."""
    if receiver is None:
        raise VtuberError(
            f"no live MVN receiver {name!r} — start one first "
            f"(nm mocap mvn listen)",
            kind="unknown_receiver")
    if getattr(receiver, "store", None) is not show.store:
        raise VtuberError(
            f"live MVN receiver {name!r} writes to another store "
            f"— it must write into show {show.name!r}'s store",
            kind="wrong_store")
    # The skeleton channel has one writer. The recorded take's
    # player holds the claim while it plays — hand the claim to
    # the live receiver, so its writes land instead of being
    # refused as a non-owner's. A channel owned by a third party
    # is refused by name, with the owner named.
    store = show.store
    if store.owner("skeleton") == show.engine:
        store.release("skeleton", show.engine)
    try:
        store.claim("skeleton", receiver.engine)
    except MocapError as exc:
        raise VtuberError(
            f"live MVN receiver {name!r} cannot take the skeleton "
            f"channel for show {show.name!r}: {exc}",
            kind="channel_busy") from exc
    show._live_mvn = {"name": name, "receiver": receiver}
    show._hold_take = True
    show._record("vtuber_live_attached", {"receiver": name})
    return {"ok": True, "show": show.name, "live": name}


def detach_live(show: Any) -> dict[str, Any]:
    """Release the live MVN receiver; the recorded take resumes."""
    live = getattr(show, "_live_mvn", None)
    if live is None:
        raise VtuberError(
            f"show {show.name!r} has no live MVN attached",
            kind="not_live")
    # Hand the skeleton channel back: the receiver's claim goes,
    # and the recorded take's claim returns, so the take player
    # writes again the moment it resumes.
    store = show.store
    receiver = live["receiver"]
    if store.owner("skeleton") == receiver.engine:
        store.release("skeleton", receiver.engine)
    if show.take is not None:
        store.claim("skeleton", show.engine)
    show._live_mvn = None
    show._hold_take = False
    show._record("vtuber_live_detached",
                 {"receiver": live["name"]})
    return {"ok": True, "show": show.name, "live": live["name"]}
