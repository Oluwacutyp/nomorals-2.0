"""The artist-IDE scene model — Devon's own ArtCraft port.

ArtCraft (storytold/artcraft, dual MIT/Apache-2.0) builds the scene
BEFORE generating: 2D compositing (layers), 3D compositing (backdrops,
foregrounds, props, depth), image-to-location (one consistent
environment across shots), character posing + camera before the first
pixel is generated, scene blocking with kitbashing, and character
identity carried across shots.

This module is that scene model, as Devon's own data: a :class:`Scene`
holds 2D :class:`Layer`\\ s, 3D :class:`StageItem`\\ s at depths, posed
:class:`Character`\\ s with identity references, one :class:`Location`,
one :class:`Camera`, and an ordered list of :class:`Shot`\\ s. The
model is pure data — no GPU, no network, no paid service — and
round-trips through JSON so a scene survives a restart.

Upstream: storytold/artcraft (MIT OR Apache-2.0). The paid provider
router, the cloud 3D projects and the dollar cost estimation are NOT
ported: Devon generates through her own rungs (see
:mod:`nomorals.media.artist.generate`), which cost nothing.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

UPSTREAM = "storytold/artcraft"
LICENCE = "MIT OR Apache-2.0"

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

LAYER_KINDS = ("image", "text", "shape", "solid")
BLEND_MODES = ("normal", "multiply", "screen")
SHAPE_KINDS = ("rect", "ellipse", "line")
SHOT_MODES = ("t2v", "i2v", "motion", "talk")

_MAX_DURATION_S = 600.0


class ArtistError(Exception):
    """A scene, compose or generation problem, named in plain words."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _is_even(value: int) -> bool:
    return value % 2 == 0


def _check_color(value: str, what: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ArtistError(f"{what} must be a #RRGGBB colour, got {value!r}")
    return value.lower()


@dataclass
class Layer:
    """One 2D compositing layer, drawn in list order (later = on top)."""

    kind: str
    name: str
    x: float = 0.0
    y: float = 0.0
    width: float = 100.0
    height: float = 100.0
    opacity: float = 1.0
    rotation: float = 0.0
    blend: str = "normal"
    # kind-specific payloads
    path: str = ""        # image
    text: str = ""        # text
    font_size: int = 24   # text
    color: str = "#ffffff"  # text / shape / solid
    shape: str = "rect"   # shape

    def validate(self) -> None:
        if self.kind not in LAYER_KINDS:
            raise ArtistError(
                f"unknown layer kind {self.kind!r} — one of "
                + ", ".join(LAYER_KINDS))
        if not self.name:
            raise ArtistError("a layer needs a name")
        if self.width <= 0 or self.height <= 0:
            raise ArtistError(
                f"layer {self.name!r} needs a positive size, got "
                f"{self.width}x{self.height}")
        if not 0.0 <= self.opacity <= 1.0:
            raise ArtistError(
                f"layer {self.name!r} opacity must be 0..1, got {self.opacity}")
        if self.blend not in BLEND_MODES:
            raise ArtistError(
                f"layer {self.name!r} blend must be one of "
                + ", ".join(BLEND_MODES))
        if self.kind == "image" and not self.path:
            raise ArtistError(f"image layer {self.name!r} needs a path")
        if self.kind == "text" and not self.text:
            raise ArtistError(f"text layer {self.name!r} needs text")
        if self.kind == "shape" and self.shape not in SHAPE_KINDS:
            raise ArtistError(
                f"shape layer {self.name!r} shape must be one of "
                + ", ".join(SHAPE_KINDS))
        if self.kind in ("text", "shape", "solid"):
            _check_color(self.color, f"layer {self.name!r} colour")
        if self.kind == "text" and self.font_size <= 0:
            raise ArtistError(
                f"text layer {self.name!r} needs a positive font size")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Layer:
        layer = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        layer.validate()
        return layer


@dataclass
class Character:
    """A character staged in the scene, with identity references.

    ``ref_image`` is a face/reference photo on disk; ``lora`` is a
    persona LoRA adapter name. Either may be empty — a character with
    neither renders as a posed mannequin placeholder in the staged
    frame (a blocking aid, never presented as the character).
    """

    name: str
    ref_image: str = ""
    lora: str = ""
    pose: str = "neutral"
    joints: dict[str, float] = field(default_factory=dict)
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0

    def validate(self) -> None:
        if not self.name:
            raise ArtistError("a character needs a name")
        if not self.pose:
            raise ArtistError(f"character {self.name!r} needs a pose name")
        if self.scale <= 0:
            raise ArtistError(
                f"character {self.name!r} scale must be positive, got {self.scale}")
        for joint, angle in self.joints.items():
            if not isinstance(angle, (int, float)):
                raise ArtistError(
                    f"character {self.name!r} joint {joint!r} angle "
                    f"must be a number, got {angle!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Character:
        character = cls(**{k: v for k, v in data.items()
                           if k in cls.__dataclass_fields__})
        character.validate()
        return character


@dataclass
class Camera:
    """The scene camera: position, orientation, field of view."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    pan: float = 0.0
    tilt: float = 0.0
    roll: float = 0.0
    fov: float = 50.0

    def validate(self) -> None:
        if not 20.0 <= self.fov <= 120.0:
            raise ArtistError(
                f"camera fov must be 20..120 degrees, got {self.fov}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Camera:
        camera = cls(**{k: v for k, v in data.items()
                        if k in cls.__dataclass_fields__})
        camera.validate()
        return camera


@dataclass
class Location:
    """One consistent environment every shot in the scene shares."""

    name: str = ""
    backdrop: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Location:
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class StageItem:
    """One 3D-staged item: an image (or a character) at a depth.

    ``depth`` is in scene units — 0 is nearest the camera, larger is
    farther. Items are drawn far-to-near so nearer items occlude
    farther ones, and nearer items project larger.
    """

    name: str
    image: str = ""
    depth: float = 1.0
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    character: str = ""

    def validate(self) -> None:
        if not self.name:
            raise ArtistError("a staged item needs a name")
        if self.depth < 0:
            raise ArtistError(
                f"staged item {self.name!r} depth must be >= 0, got {self.depth}")
        if self.scale <= 0:
            raise ArtistError(
                f"staged item {self.name!r} scale must be positive, got {self.scale}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageItem:
        item = cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})
        item.validate()
        return item


@dataclass
class Shot:
    """One shot of the film: a prompt, a mode, a duration, a voice.

    Modes: ``t2v`` (text to video), ``i2v`` (staged frame to video),
    ``motion`` (staged frame animated by the motion studio),
    ``talk`` (the named character speaks ``lines`` — lip-synced).
    """

    name: str
    prompt: str
    mode: str = "t2v"
    duration_s: float = 5.0
    lines: str = ""
    character: str = ""
    image: str = ""
    camera: dict[str, Any] | None = None

    def validate(self, known_characters: set[str] | None = None) -> None:
        if not self.name:
            raise ArtistError("a shot needs a name")
        if not self.prompt:
            raise ArtistError(f"shot {self.name!r} needs a prompt")
        if self.mode not in SHOT_MODES:
            raise ArtistError(
                f"shot {self.name!r} mode must be one of "
                + ", ".join(SHOT_MODES))
        if not 0.0 < self.duration_s <= _MAX_DURATION_S:
            raise ArtistError(
                f"shot {self.name!r} duration must be 0..{_MAX_DURATION_S:g}s, "
                f"got {self.duration_s}")
        if self.mode == "talk" and not self.lines:
            raise ArtistError(
                f"talk shot {self.name!r} needs lines for the character to say")
        if self.character and known_characters is not None \
                and self.character not in known_characters:
            raise ArtistError(
                f"shot {self.name!r} names unknown character "
                f"{self.character!r}")
        if self.camera is not None:
            Camera.from_dict(self.camera)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Shot:
        shot = cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})
        return shot


@dataclass
class Scene:
    """A whole artist-IDE scene: layers, staging, characters, shots."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    layers: list[Layer] = field(default_factory=list)
    characters: list[Character] = field(default_factory=list)
    staging: list[StageItem] = field(default_factory=list)
    location: Location = field(default_factory=Location)
    camera: Camera = field(default_factory=Camera)
    shots: list[Shot] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    backdrop_color: str = "#1a1a2e"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    upstream: str = UPSTREAM
    licence: str = LICENCE

    def __post_init__(self) -> None:
        if not self.title:
            raise ArtistError("a scene needs a title")
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise ArtistError("scene size must be whole pixels")
        if self.width < 16 or self.height < 16:
            raise ArtistError(
                f"scene size must be at least 16x16, got {self.width}x{self.height}")
        if not _is_even(self.width) or not _is_even(self.height):
            raise ArtistError(
                f"scene size must be even (encoders need it), got "
                f"{self.width}x{self.height}")
        _check_color(self.backdrop_color, "scene backdrop colour")
        for layer in self.layers:
            layer.validate()
        for character in self.characters:
            character.validate()
        for item in self.staging:
            item.validate()
        self.camera.validate()
        known = {c.name for c in self.characters}
        for shot in self.shots:
            shot.validate(known)

    # -- layers ----------------------------------------------------
    def layer(self, name: str) -> Layer:
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise ArtistError(f"no layer named {name!r} in scene {self.title!r}")

    def add_layer(self, *, kind: str, name: str, **kw: Any) -> Layer:
        if any(layer.name == name for layer in self.layers):
            raise ArtistError(f"scene already has a layer named {name!r}")
        layer = Layer(kind=kind, name=name, **kw)
        layer.validate()
        self.layers.append(layer)
        self._touch()
        return layer

    def move_layer(self, name: str, index: int) -> None:
        layer = self.layer(name)
        self.layers.remove(layer)
        self.layers.insert(max(0, min(index, len(self.layers))), layer)
        self._touch()

    def remove_layer(self, name: str) -> Layer:
        layer = self.layer(name)
        self.layers.remove(layer)
        self._touch()
        return layer

    # -- characters ------------------------------------------------
    def character(self, name: str) -> Character:
        for character in self.characters:
            if character.name == name:
                return character
        raise ArtistError(
            f"no character named {name!r} in scene {self.title!r}")

    def add_character(self, name: str, **kw: Any) -> Character:
        if any(c.name == name for c in self.characters):
            raise ArtistError(f"scene already has a character named {name!r}")
        character = Character(name=name, **kw)
        character.validate()
        self.characters.append(character)
        self._touch()
        return character

    def pose_character(self, name: str, pose: str,
                       joints: dict[str, float] | None = None) -> Character:
        character = self.character(name)
        character.pose = pose
        if joints:
            character.joints.update(joints)
        character.validate()
        self._touch()
        return character

    # -- camera / location / staging -------------------------------
    def set_camera(self, **kw: Any) -> Camera:
        values = asdict(self.camera)
        values.update(kw)
        self.camera = Camera.from_dict(values)
        self._touch()
        return self.camera

    def set_location(self, name: str, backdrop: str = "",
                     notes: str = "") -> Location:
        self.location = Location(name=name, backdrop=backdrop, notes=notes)
        self._touch()
        return self.location

    def stage_item(self, name: str, **kw: Any) -> StageItem:
        if any(item.name == name for item in self.staging):
            raise ArtistError(f"scene already stages an item named {name!r}")
        item = StageItem(name=name, **kw)
        item.validate()
        if item.character:
            self.character(item.character)  # raises when unknown
        self.staging.append(item)
        self._touch()
        return item

    # -- shots -----------------------------------------------------
    def shot(self, name: str) -> Shot:
        for shot in self.shots:
            if shot.name == name:
                return shot
        raise ArtistError(f"no shot named {name!r} in scene {self.title!r}")

    def add_shot(self, name: str, *, prompt: str, **kw: Any) -> Shot:
        if any(shot.name == name for shot in self.shots):
            raise ArtistError(f"scene already has a shot named {name!r}")
        shot = Shot(name=name, prompt=prompt, **kw)
        shot.validate({c.name for c in self.characters})
        self.shots.append(shot)
        self._touch()
        return shot

    # -- serialisation ---------------------------------------------
    def _touch(self) -> None:
        self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "layers": [layer.to_dict() for layer in self.layers],
            "characters": [c.to_dict() for c in self.characters],
            "staging": [item.to_dict() for item in self.staging],
            "location": self.location.to_dict(),
            "camera": self.camera.to_dict(),
            "shots": [shot.to_dict() for shot in self.shots],
            "artifacts": self.artifacts,
            "backdrop_color": self.backdrop_color,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "upstream": self.upstream,
            "licence": self.licence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scene:
        if not isinstance(data, dict) or not data.get("id") \
                or not data.get("title"):
            raise ArtistError("a stored scene needs an id and a title")
        try:
            return cls(
                id=str(data["id"]),
                title=str(data["title"]),
                width=int(data.get("width", DEFAULT_WIDTH)),
                height=int(data.get("height", DEFAULT_HEIGHT)),
                layers=[Layer.from_dict(item)
                        for item in data.get("layers", [])],
                characters=[Character.from_dict(item)
                            for item in data.get("characters", [])],
                staging=[StageItem.from_dict(item)
                         for item in data.get("staging", [])],
                location=Location.from_dict(data.get("location", {}) or {}),
                camera=Camera.from_dict(data.get("camera", {}) or {}),
                shots=[Shot.from_dict(item)
                       for item in data.get("shots", [])],
                artifacts=dict(data.get("artifacts", {}) or {}),
                backdrop_color=str(data.get("backdrop_color", "#1a1a2e")),
                created_at=str(data.get("created_at", _now())),
                updated_at=str(data.get("updated_at", _now())),
                upstream=str(data.get("upstream", UPSTREAM)),
                licence=str(data.get("licence", LICENCE)),
            )
        except ArtistError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise ArtistError(f"stored scene is malformed: {exc}") from exc
