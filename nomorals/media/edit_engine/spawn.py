"""Per-scene actor spawning for Scene3D documents.

The shape is mined from storytold/ActorSpawningPlugin (PATTERN ONLY,
no code ported — no licence): one spawner object with typed spawn
verbs per kind (static mesh, skeletal mesh, sound), placement resolved
against the player's location, and a destroy timer so actors do not
pile up unused. Devon's version keeps the shape and fixes what the
clone admits is broken:

* Placement is deliberate, not random. The clone scatters actors at
  a random offset around the player and carries a TODO — "Put this
  in front of the camera, or on the (camera -> player) arc!" — so
  the four placement modes here are ``at`` (an explicit position),
  ``offset`` (from a named anchor node), ``in_front_of`` (on the
  camera -> anchor arc, the clone's TODO made real) and
  ``random_around`` (the clone's behaviour, kept as an explicit
  opt-in with stated bounds and a seed).
* The clone tracks nothing: a spawned actor is a bare pointer, the
  static-mesh verb even carries a "This will memory leak" TODO, and
  the only lifecycle is a 100-second destroy timer. Here every actor
  has a real lifecycle — ``spawned`` -> ``active`` -> ``despawned``
  — recorded per actor, and despawning removes the node AND any
  animation tracks targeting it (a leftover track breaks scene
  validation), so the slot is genuinely freed.
* The clone has no cap ("they may pile up unused"). Here each scene
  carries its own budget, enforced at the scene.

No central registry, no shared queue: a ``SceneSpawner`` is bound
to ONE ``Scene3D`` document. Two scenes spawn independently; nothing
funnels through a centre (the owner's standing law — a centre
becomes a queue).
"""

from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..mocap.bvh import parse_bvh, skeleton_to_dict
from .scene3d import (
    Light,
    Mesh,
    Rotation,
    Scene3D,
    Scene3DError,
    SceneNode,
    Transform,
)


class SpawnError(Scene3DError):
    """A spawn request that cannot be honoured, naming the cause."""


class SpawnBudgetError(SpawnError):
    """The scene's actor budget is exhausted."""


#: Actor kinds, mirroring the clone's three spawn verbs plus the
#: light payload a Scene3D node already carries.
SPAWN_KINDS = ("prop", "character", "sound", "light")

#: Primitive stand-in bodies for a character (the skeleton is the
#: real data; the mesh is a body until a rig consumes the skeleton).
CHARACTER_BODY = "cylinder"

#: Default per-scene actor budget.
DEFAULT_MAX_ACTORS = 32

#: Default lifetime for a spawned actor when the request names none
#: (``None`` = lives until despawned). The clone's destroy timer was
#: a flat 100 seconds for mesh/skeletal actors.
DEFAULT_LIFETIME_S: float | None = None


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------

#: Placement modes. ``at`` is an explicit position; ``offset`` is
#: relative to a named anchor node (or the active camera); ``in_front_of``
#: puts the actor on the camera -> anchor arc; ``random_around`` is the
#: clone's scatter, bounds stated, seed optional.
PLACEMENT_MODES = ("at", "offset", "in_front_of", "random_around")


def _vec3(value: Any, what: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise SpawnError(f"{what} must be a 3-number position, "
                         f"got {value!r}")
    vals = list(value)
    if len(vals) != 3:
        raise SpawnError(f"{what} must be a 3-number position, "
                         f"got {value!r}")
    try:
        out = (float(vals[0]), float(vals[1]), float(vals[2]))
    except (TypeError, ValueError):
        raise SpawnError(f"{what} must be numbers, got {value!r}") from None
    if any(math.isnan(v) or math.isinf(v) for v in out):
        raise SpawnError(f"{what} must be finite, got {value!r}")
    return out


@dataclass
class Placement:
    """Where a spawned actor lands, as data.

    ``mode`` is one of :data:`PLACEMENT_MODES`. ``position`` is the
    explicit point (``at``) or the offset (``offset`` /
    ``in_front_of`` uses ``distance`` instead). ``anchor`` names a
    scene node, or ``"camera"`` for the active camera. ``bounds``
    is ``{"x": [lo, hi], "y": [...], "z": [...]}`` for
    ``random_around``. ``seed`` makes the scatter deterministic.
    """

    mode: str = "at"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    anchor: str | None = None
    distance: float = 2.0
    bounds: dict[str, list[float]] | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in PLACEMENT_MODES:
            raise SpawnError(
                f"placement mode must be one of {PLACEMENT_MODES}, "
                f"got {self.mode!r}")
        self.position = _vec3(self.position, "placement position")
        if self.anchor is not None:
            self.anchor = str(self.anchor)
        self.distance = float(self.distance)
        if self.mode == "in_front_of" and self.distance <= 0:
            raise SpawnError(
                f"in_front_of distance must be > 0, got {self.distance}")
        if self.bounds is not None:
            clean: dict[str, list[float]] = {}
            for axis in ("x", "y", "z"):
                raw = self.bounds.get(axis)
                if raw is None:
                    continue
                if len(raw) != 2:
                    raise SpawnError(
                        f"random_around bounds[{axis!r}] needs [lo, hi], "
                        f"got {raw!r}")
                lo, hi = float(raw[0]), float(raw[1])
                if lo > hi:
                    raise SpawnError(
                        f"random_around bounds[{axis!r}] needs lo <= hi, "
                        f"got [{lo}, {hi}]")
                clean[axis] = [lo, hi]
            self.bounds = clean
        if self.seed is not None:
            self.seed = int(self.seed)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode,
                "position": list(self.position),
                "anchor": self.anchor,
                "distance": self.distance,
                "bounds": copy.deepcopy(self.bounds),
                "seed": self.seed}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Placement:
        if not isinstance(data, dict):
            raise SpawnError(
                f"placement must be a dict, got {data!r}")
        return cls(mode=str(data.get("mode", "at")),
                   position=_vec3(data.get("position", (0, 0, 0)),
                                  "placement position"),
                   anchor=data.get("anchor"),
                   distance=float(data.get("distance", 2.0)),
                   bounds=copy.deepcopy(data.get("bounds")),
                   seed=data.get("seed"))

    @classmethod
    def at(cls, position: Any) -> Placement:
        return cls(mode="at", position=_vec3(position, "position"))

    @classmethod
    def offset_from(cls, anchor: str,
                    offset: Any) -> Placement:
        return cls(mode="offset", anchor=str(anchor),
                   position=_vec3(offset, "offset"))

    @classmethod
    def in_front_of(cls, anchor: str,
                    distance: float = 2.0) -> Placement:
        return cls(mode="in_front_of", anchor=str(anchor),
                   distance=float(distance))

    @classmethod
    def random_around(
            cls, anchor: str | None = None, *,
            bounds: dict[str, list[float]] | None = None,
            seed: int | None = None) -> Placement:
        return cls(mode="random_around", anchor=anchor, bounds=bounds,
                   seed=seed)


# ---------------------------------------------------------------------------
# spawn request
# ---------------------------------------------------------------------------

@dataclass
class SpawnRequest:
    """One spawn, as data. ``name`` may be omitted — the spawner
    auto-names it ``spawn-<kind>-<n>`` inside the scene.

    ``kind`` is one of :data:`SPAWN_KINDS`: ``prop`` is a mesh
    primitive (``primitive`` + numeric ``params`` + optional
    ``material`` naming a scene material); ``character`` is a
    skeletal actor — ``bvh_path`` (a BVH file, parsed through the
    mocap organ) or ``skeleton`` (a skeleton dict) carries the
    joint hierarchy, the mesh is a stand-in body; ``sound`` is a
    positional sound (``sound_clip`` names the audio, ``volume``
    0..1, ``destroy_after_play`` despawns it when its
    ``duration_s`` elapses); ``light`` is a scene light
    (``light_kind`` / ``color`` / ``intensity``).

    ``lifetime_s`` is the destroy timer in seconds (the clone's
    100-second timer, made explicit and per-actor); ``None`` lives
    until despawned. ``parent`` names a scene node to parent under.
    ``rotation`` is euler degrees; ``look_at`` aims -Z at a point.
    ``scale`` is the node scale.
    """

    kind: str
    placement: Placement = field(default_factory=Placement)
    name: str | None = None
    lifetime_s: float | None = DEFAULT_LIFETIME_S
    parent: str | None = None
    rotation: tuple[float, float, float] | None = None
    look_at: tuple[float, float, float] | None = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    primitive: str = "cube"
    params: dict[str, float] = field(default_factory=dict)
    material: str | None = None
    bvh_path: str | None = None
    skeleton: dict[str, Any] | None = None
    sound_clip: str | None = None
    volume: float = 1.0
    destroy_after_play: bool = False
    duration_s: float | None = None
    light_kind: str = "point"
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in SPAWN_KINDS:
            raise SpawnError(
                f"spawn kind must be one of {SPAWN_KINDS}, "
                f"got {self.kind!r}")
        if not isinstance(self.placement, Placement):
            self.placement = Placement.from_dict(dict(self.placement))
        if self.name is not None:
            self.name = str(self.name)
            if not self.name.strip():
                raise SpawnError("spawn name must not be blank")
        if self.lifetime_s is not None:
            self.lifetime_s = float(self.lifetime_s)
            if self.lifetime_s <= 0:
                raise SpawnError(
                    f"lifetime_s must be > 0, got {self.lifetime_s}")
        if self.parent is not None:
            self.parent = str(self.parent)
        if self.rotation is not None:
            self.rotation = _vec3(self.rotation, "rotation")
        if self.look_at is not None:
            self.look_at = _vec3(self.look_at, "look_at")
        self.scale = _vec3(self.scale, "scale")
        if any(s <= 0 for s in self.scale):
            raise SpawnError(
                f"scale components must be > 0, got {self.scale}")
        if self.kind == "prop":
            if self.primitive not in ("cube", "sphere", "plane",
                                      "cylinder", "cone"):
                raise SpawnError(
                    f"prop primitive must be one of cube/sphere/plane/"
                    f"cylinder/cone, got {self.primitive!r}")
            clean: dict[str, float] = {}
            for key, value in (self.params or {}).items():
                if isinstance(value, bool) \
                        or not isinstance(value, (int, float)):
                    raise SpawnError(
                        f"primitive param {key!r} must be a number, "
                        f"got {value!r}")
                clean[str(key)] = float(value)
            self.params = clean
        if self.kind == "character":
            if self.bvh_path is None and self.skeleton is None:
                raise SpawnError(
                    "character spawn needs bvh_path or skeleton — "
                    "a skeletal actor without joints is a prop")
            if self.bvh_path is not None and self.skeleton is not None:
                raise SpawnError(
                    "character spawn takes bvh_path OR skeleton, "
                    "not both")
        if self.kind == "sound":
            if not self.sound_clip:
                raise SpawnError(
                    "sound spawn needs sound_clip — a sound actor "
                    "with no audio is silent")
            self.sound_clip = str(self.sound_clip)
            self.volume = float(self.volume)
            if not 0.0 <= self.volume <= 1.0:
                raise SpawnError(
                    f"sound volume must be in [0, 1], got {self.volume}")
            if self.duration_s is not None:
                self.duration_s = float(self.duration_s)
                if self.duration_s <= 0:
                    raise SpawnError(
                        f"duration_s must be > 0, got {self.duration_s}")
        if self.kind == "light":
            if self.light_kind not in ("ambient", "directional",
                                       "point", "spot"):
                raise SpawnError(
                    f"light kind must be one of ambient/directional/"
                    f"point/spot, got {self.light_kind!r}")
            self.color = _vec3(self.color, "light color")
            if any(c < 0.0 for c in self.color):
                raise SpawnError(
                    f"light color components must be >= 0, "
                    f"got {self.color}")
            self.intensity = float(self.intensity)
            if self.intensity < 0.0:
                raise SpawnError(
                    f"light intensity must be >= 0, "
                    f"got {self.intensity}")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind,
                "placement": self.placement.to_dict(),
                "name": self.name,
                "lifetime_s": self.lifetime_s,
                "parent": self.parent,
                "rotation": list(self.rotation)
                if self.rotation is not None else None,
                "look_at": list(self.look_at)
                if self.look_at is not None else None,
                "scale": list(self.scale),
                "primitive": self.primitive,
                "params": dict(self.params),
                "material": self.material,
                "bvh_path": self.bvh_path,
                "skeleton": copy.deepcopy(self.skeleton),
                "sound_clip": self.sound_clip,
                "volume": self.volume,
                "destroy_after_play": self.destroy_after_play,
                "duration_s": self.duration_s,
                "light_kind": self.light_kind,
                "color": list(self.color),
                "intensity": self.intensity}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpawnRequest:
        if not isinstance(data, dict):
            raise SpawnError(
                f"spawn request must be a dict, got {data!r}")
        return cls(kind=str(data.get("kind", "")),
                   placement=Placement.from_dict(
                       dict(data.get("placement") or {})),
                   name=data.get("name"),
                   lifetime_s=data.get("lifetime_s"),
                   parent=data.get("parent"),
                   rotation=data.get("rotation"),
                   look_at=data.get("look_at"),
                   scale=_vec3(data.get("scale", (1, 1, 1)), "scale"),
                   primitive=str(data.get("primitive", "cube")),
                   params=dict(data.get("params") or {}),
                   material=data.get("material"),
                   bvh_path=data.get("bvh_path"),
                   skeleton=copy.deepcopy(data.get("skeleton")),
                   sound_clip=data.get("sound_clip"),
                   volume=float(data.get("volume", 1.0)),
                   destroy_after_play=bool(
                       data.get("destroy_after_play", False)),
                   duration_s=data.get("duration_s"),
                   light_kind=str(data.get("light_kind", "point")),
                   color=_vec3(data.get("color", (1, 1, 1)),
                               "light color"),
                   intensity=float(data.get("intensity", 1.0)))


# ---------------------------------------------------------------------------
# actor state
# ---------------------------------------------------------------------------

#: Lifecycle states. ``spawned`` is the moment the node lands in
#: the scene; ``active`` is after the spawner's first tick over it;
#: ``despawned`` is after removal (the record stays queryable).
ACTOR_STATUSES = ("spawned", "active", "despawned")


@dataclass
class ActorState:
    """One spawned actor, as the scene can query it."""

    name: str
    kind: str
    status: str
    position: tuple[float, float, float]
    spawned_at: float
    lifetime_s: float | None = None
    despawned_at: float | None = None
    despawn_reason: str | None = None

    def age_s(self, now: float | None = None) -> float:
        end = self.despawned_at if self.status == "despawned" \
            else (now if now is not None else time.time())
        return max(0.0, end - self.spawned_at)

    def remaining_s(self, now: float | None = None) -> float | None:
        if self.lifetime_s is None:
            return None
        return max(0.0, self.lifetime_s - self.age_s(now))

    def to_dict(self) -> dict[str, Any]:
        now = time.time()
        return {"name": self.name, "kind": self.kind,
                "status": self.status,
                "position": list(self.position),
                "age_s": round(self.age_s(now), 3),
                "remaining_s": (round(self.remaining_s(now), 3)
                                if self.remaining_s(now) is not None
                                else None),
                "lifetime_s": self.lifetime_s,
                "despawned_at": self.despawned_at,
                "despawn_reason": self.despawn_reason}


# ---------------------------------------------------------------------------
# the spawner — one per scene, never shared
# ---------------------------------------------------------------------------

class SceneSpawner:
    """Spawn actors into ONE ``Scene3D`` document.

    The spawner owns nothing outside its scene: the budget is
    enforced here, the actor records live here, and despawning
    frees the scene slot here. Two scenes spawn independently —
    there is no central registry and no shared queue.
    """

    def __init__(self, scene: Scene3D, *,
                 max_actors: int = DEFAULT_MAX_ACTORS) -> None:
        if not isinstance(scene, Scene3D):
            raise SpawnError(
                f"SceneSpawner needs a Scene3D, got {type(scene).__name__}")
        self.scene = scene
        self.max_actors = int(max_actors)
        if self.max_actors <= 0:
            raise SpawnError(
                f"max_actors must be > 0, got {max_actors}")
        self._actors: dict[str, ActorState] = {}
        self._seq = 0

    # -- construction ------------------------------------------------
    @classmethod
    def from_scene(cls, scene: Scene3D, *,
                   max_actors: int = DEFAULT_MAX_ACTORS) -> SceneSpawner:
        """Rebuild a spawner for a loaded scene: nodes carrying a
        ``spawn`` provenance block in ``extras`` become live actors
        again (status ``spawned`` until the first tick)."""
        spawner = cls(scene, max_actors=max_actors)
        for node in scene.nodes:
            prov = (node.extras or {}).get("spawn")
            if not isinstance(prov, dict):
                continue
            spawned_at = float(prov.get("spawned_at", time.time()))
            spawner._actors[node.name] = ActorState(
                name=node.name, kind=str(prov.get("kind", "prop")),
                status="spawned",
                position=tuple(node.transform.position),
                spawned_at=spawned_at,
                lifetime_s=prov.get("lifetime_s"))
        spawner._seq = len(spawner._actors)
        return spawner

    # -- queries -------------------------------------------------------
    @property
    def alive_count(self) -> int:
        return sum(1 for a in self._actors.values()
                   if a.status != "despawned")

    @property
    def budget(self) -> dict[str, Any]:
        return {"max_actors": self.max_actors,
                "alive": self.alive_count,
                "free": self.max_actors - self.alive_count}

    def state(self, name: str) -> ActorState:
        try:
            return self._actors[str(name)]
        except KeyError:
            raise SpawnError(
                f"no spawned actor {name!r} in scene "
                f"{self.scene.title!r} — spawned actors: "
                f"{', '.join(sorted(self._actors)) or 'none'}") from None

    def list_actors(self, status: str | None = None) -> list[ActorState]:
        if status is not None and status not in ACTOR_STATUSES:
            raise SpawnError(
                f"actor status must be one of {ACTOR_STATUSES}, "
                f"got {status!r}")
        return [a for a in self._actors.values()
                if status is None or a.status == status]

    # -- placement -----------------------------------------------------
    def _anchor_position(self, anchor: str | None) -> tuple[float, float, float]:
        if anchor is None:
            return (0.0, 0.0, 0.0)
        if anchor == "camera":
            return tuple(self.scene.camera().transform.position)
        try:
            node = self.scene.node(anchor)
        except Scene3DError:
            raise SpawnError(
                f"placement anchor {anchor!r} is not a scene node — "
                f"scene has: "
                f"{', '.join(n.name for n in self.scene.nodes) or 'none'}"
                ) from None
        return tuple(node.transform.position)

    def _resolve(self, placement: Placement) -> tuple[float, float, float]:
        if placement.mode == "at":
            return placement.position
        if placement.mode == "offset":
            ax, ay, az = self._anchor_position(placement.anchor)
            px, py, pz = placement.position
            return (ax + px, ay + py, az + pz)
        if placement.mode == "in_front_of":
            # On the camera -> anchor arc: the anchor's position
            # pushed away from the camera along the camera -> anchor
            # direction. The clone's TODO, made real.
            try:
                cam_pos = tuple(self.scene.camera().transform.position)
            except Scene3DError:
                raise SpawnError(
                    "in_front_of placement needs a scene camera — "
                    "this scene has none") from None
            anchor_pos = self._anchor_position(placement.anchor)
            dx = anchor_pos[0] - cam_pos[0]
            dy = anchor_pos[1] - cam_pos[1]
            dz = anchor_pos[2] - cam_pos[2]
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            if length == 0.0:
                raise SpawnError(
                    "in_front_of placement needs the anchor away "
                    "from the camera — they share a position")
            ux, uy, uz = dx / length, dy / length, dz / length
            d = placement.distance
            return (anchor_pos[0] + ux * d, anchor_pos[1] + uy * d,
                    anchor_pos[2] + uz * d)
        # random_around — the clone's scatter, bounds stated.
        ax, ay, az = self._anchor_position(placement.anchor)
        bounds = placement.bounds or {}
        rng = random.Random(placement.seed)
        def _pick(axis: str, default: tuple[float, float]) -> float:
            lo, hi = bounds.get(axis, list(default))
            return rng.uniform(float(lo), float(hi))
        return (ax + _pick("x", (-4.0, 4.0)),
                ay + _pick("y", (-4.0, 4.0)),
                az + _pick("z", (0.5, 2.0)))

    # -- spawning ------------------------------------------------------
    def _next_name(self, kind: str) -> str:
        while True:
            self._seq += 1
            name = f"spawn-{kind}-{self._seq}"
            if name not in self._actors \
                    and all(n.name != name for n in self.scene.nodes):
                return name

    def spawn(self, request: SpawnRequest,
              now: float | None = None) -> ActorState:
        """Spawn one actor into the scene. Returns its state.

        Refuses, naming the cause: unknown kind/placement/anchor/
        parent/material, a duplicate name, or the budget.
        """
        if not isinstance(request, SpawnRequest):
            request = SpawnRequest.from_dict(dict(request))
        if self.alive_count >= self.max_actors:
            raise SpawnBudgetError(
                f"scene {self.scene.title!r} already has "
                f"{self.alive_count} actors (cap {self.max_actors}) — "
                f"despawn one before spawning another")
        name = request.name or self._next_name(request.kind)
        if name in self._actors \
                and self._actors[name].status != "despawned":
            raise SpawnError(
                f"scene {self.scene.title!r} already has a live "
                f"actor named {name!r}")
        if any(n.name == name for n in self.scene.nodes):
            raise SpawnError(
                f"scene {self.scene.title!r} already has a node "
                f"named {name!r}")
        if request.parent is not None:
            try:
                self.scene.node(request.parent)
            except Scene3DError:
                raise SpawnError(
                    f"spawn parent {request.parent!r} is not a scene "
                    f"node — scene has: "
                    f"{', '.join(n.name for n in self.scene.nodes) or 'none'}"
                    ) from None
        if request.material is not None \
                and request.material not in self.scene.materials:
            raise SpawnError(
                f"spawn material {request.material!r} is not a scene "
                f"material — scene has: "
                f"{', '.join(sorted(self.scene.materials)) or 'none'}")
        position = self._resolve(request.placement)
        rotation = Rotation.look_at(request.look_at) \
            if request.look_at is not None else Rotation.from_euler(
                *(request.rotation or (0.0, 0.0, 0.0)))
        node = self._build_node(request, name, position, rotation)
        spawned_at = now if now is not None else time.time()
        node.extras["spawn"] = {
            "kind": request.kind,
            "spawned_at": spawned_at,
            "lifetime_s": request.lifetime_s,
            "request": request.to_dict(),
        }
        self.scene.nodes.append(node)
        try:
            self.scene._validate()
        except Scene3DError:
            self.scene.nodes.pop()
            raise
        self.scene.seal()
        state = ActorState(name=name, kind=request.kind,
                           status="spawned", position=position,
                           spawned_at=spawned_at,
                           lifetime_s=request.lifetime_s)
        self._actors[name] = state
        return state

    def _build_node(self, request: SpawnRequest, name: str,
                    position: tuple[float, float, float],
                    rotation: Rotation) -> SceneNode:
        transform = Transform(position=position, rotation=rotation,
                              scale=request.scale)
        if request.kind == "prop":
            return SceneNode(
                name=name, transform=transform,
                parent=request.parent,
                mesh=Mesh(kind="primitive",
                          primitive=request.primitive,
                          params=dict(request.params)),
                material=request.material)
        if request.kind == "character":
            skeleton = self._load_skeleton(request)
            return SceneNode(
                name=name, transform=transform,
                parent=request.parent,
                mesh=Mesh(kind="primitive",
                          primitive=CHARACTER_BODY,
                          params={"radius": 0.3, "height": 1.7}),
                material=request.material,
                extras={"skeleton": skeleton,
                        "skeleton_source": (
                            request.bvh_path or "inline")})
        if request.kind == "sound":
            return SceneNode(
                name=name, transform=transform,
                parent=request.parent,
                extras={"sound": {
                    "clip": request.sound_clip,
                    "volume": request.volume,
                    "destroy_after_play": request.destroy_after_play,
                    "duration_s": request.duration_s}})
        # light
        return SceneNode(
            name=name, transform=transform, parent=request.parent,
            light=Light(kind=request.light_kind, color=request.color,
                        intensity=request.intensity))

    @staticmethod
    def _load_skeleton(request: SpawnRequest) -> dict[str, Any]:
        if request.skeleton is not None:
            return copy.deepcopy(request.skeleton)
        path = Path(os.path.expanduser(str(request.bvh_path)))
        if not path.is_file():
            raise SpawnError(
                f"character BVH file not found: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SpawnError(
                f"could not read character BVH {path}: {exc}") from exc
        try:
            skeleton, _frames = parse_bvh(text)
        except Exception as exc:  # noqa: BLE001 - BVH parse is guarded
            raise SpawnError(
                f"character BVH {path} does not parse: {exc}") from exc
        return skeleton_to_dict(skeleton)

    # -- lifecycle -----------------------------------------------------
    def despawn(self, name: str, *, reason: str = "explicit",
                now: float | None = None) -> ActorState:
        """Remove an actor and free its scene slot: the node goes,
        and so does every animation track targeting it (a leftover
        track breaks scene validation). The record stays queryable
        as ``despawned``."""
        state = self.state(name)
        if state.status == "despawned":
            raise SpawnError(f"actor {name!r} is already despawned")
        moment = now if now is not None else time.time()
        self.scene.nodes = [n for n in self.scene.nodes
                            if n.name != name]
        prefix = f"nodes/{name}/"
        self.scene.tracks = [t for t in self.scene.tracks
                             if not str(t.target).startswith(prefix)]
        self.scene._validate()
        self.scene.seal()
        state.status = "despawned"
        state.despawned_at = moment
        state.despawn_reason = reason
        return state

    def tick(self, now: float | None = None) -> list[str]:
        """One update pass: ``spawned`` actors become ``active``,
        and expired ones are culled — the destroy timer (lifetime
        elapsed) or a sound actor whose clip finished
        (``destroy_after_play`` with ``duration_s`` elapsed).
        Returns the names despawned by this tick."""
        moment = now if now is not None else time.time()
        culled: list[str] = []
        for actor in list(self._actors.values()):
            if actor.status == "despawned":
                continue
            if actor.status == "spawned":
                actor.status = "active"
            expired = False
            reason = ""
            if actor.lifetime_s is not None \
                    and actor.age_s(moment) >= actor.lifetime_s:
                expired, reason = True, "lifetime expired"
            if not expired and actor.kind == "sound":
                node = next((n for n in self.scene.nodes
                             if n.name == actor.name), None)
                sound = (node.extras or {}).get("sound") \
                    if node is not None else None
                if isinstance(sound, dict) \
                        and sound.get("destroy_after_play") \
                        and sound.get("duration_s") is not None \
                        and actor.age_s(moment) >= float(
                            sound["duration_s"]):
                    expired, reason = True, "sound finished"
            if expired:
                self.despawn(actor.name, reason=reason, now=moment)
                culled.append(actor.name)
        return culled


# ---------------------------------------------------------------------------
# file operations — load a scene document, spawn, save it sealed
# ---------------------------------------------------------------------------

def _load(path: str | os.PathLike) -> Scene3D:
    try:
        return Scene3D.load(path)
    except Scene3DError:
        raise
    except Exception as exc:  # noqa: BLE001 - load is guarded
        raise SpawnError(
            f"could not load scene {path}: {exc}") from exc


def spawn_into_file(scene_path: str | os.PathLike,
                    request: SpawnRequest | dict[str, Any], *,
                    max_actors: int = DEFAULT_MAX_ACTORS,
                    now: float | None = None) -> dict[str, Any]:
    """Load a scene3d document, spawn one actor, save it sealed.
    Returns the actor state plus the scene path and budget."""
    scene = _load(scene_path)
    spawner = SceneSpawner.from_scene(scene, max_actors=max_actors)
    state = spawner.spawn(request, now=now)
    scene.save(scene_path)
    return {"actor": state.to_dict(), "scene": str(scene_path),
            "budget": spawner.budget}


def despawn_from_file(scene_path: str | os.PathLike, name: str, *,
                      reason: str = "explicit",
                      now: float | None = None) -> dict[str, Any]:
    """Load a scene3d document, despawn one actor, save it sealed."""
    scene = _load(scene_path)
    spawner = SceneSpawner.from_scene(scene)
    state = spawner.despawn(name, reason=reason, now=now)
    scene.save(scene_path)
    return {"actor": state.to_dict(), "scene": str(scene_path),
            "budget": spawner.budget}


def actors_in_file(scene_path: str | os.PathLike, *,
                   status: str | None = None) -> dict[str, Any]:
    """Spawned actors in a scene3d document, with the budget."""
    scene = _load(scene_path)
    spawner = SceneSpawner.from_scene(scene)
    return {"scene": str(scene_path),
            "actors": [a.to_dict()
                       for a in spawner.list_actors(status)],
            "budget": spawner.budget}


def tick_file(scene_path: str | os.PathLike, *,
              now: float | None = None) -> dict[str, Any]:
    """Run one spawner tick over a scene3d document (activates new
    actors, culls expired ones) and save it sealed if anything
    changed."""
    scene = _load(scene_path)
    spawner = SceneSpawner.from_scene(scene)
    culled = spawner.tick(now=now)
    # Actor status lives in the spawner's memory, not the document —
    # only a cull changes the file, so only a cull saves it.
    if culled:
        scene.save(scene_path)
    return {"scene": str(scene_path), "culled": culled,
            "actors": [a.to_dict() for a in spawner.list_actors()],
            "budget": spawner.budget}


__all__ = [
    "ACTOR_STATUSES",
    "CHARACTER_BODY",
    "DEFAULT_LIFETIME_S",
    "DEFAULT_MAX_ACTORS",
    "PLACEMENT_MODES",
    "SPAWN_KINDS",
    "ActorState",
    "Placement",
    "SceneSpawner",
    "SpawnBudgetError",
    "SpawnError",
    "SpawnRequest",
    "actors_in_file",
    "despawn_from_file",
    "spawn_into_file",
    "tick_file",
]
