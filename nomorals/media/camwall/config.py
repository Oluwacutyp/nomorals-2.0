"""Cameras-wall config — the viewer's camera JSON, in Python.

The clone's config (example.json) is a ``{"cameras": [...]}`` document.
Each camera carries EITHER a ``full-url`` OR a broken-down host +
image path (``remote-host`` / ``local-host`` + ``image``), plus
``name``, ``location``, ``vendor`` and ``type``. An optional ``mjpeg``
path (or ``mjpeg-full-url``) names the MJPEG stream for the same
camera. Real IP cameras need credentials, which the clone never had
fields for — this port adds optional ``username`` / ``password`` keys,
the shape's extension, not a replacement.

The default config home is ``~/.nomorals/camwall/cameras.json`` —
the same ``Path.home() / ".nomorals"`` shape the smarthome gate
uses. A wall loads from a dict, a JSON file, or a path; it never
guesses a camera.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import CamWallError

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "CameraConfig",
    "cameras_from_json",
    "dump_config",
    "load_config",
]


#: Where a wall's cameras live when no path is given.
DEFAULT_CONFIG_PATH = Path.home() / ".nomorals" / "camwall" / "cameras.json"


@dataclass
class CameraConfig:
    """One camera, as the viewer configures it."""

    name: str = ""
    location: str = ""
    vendor: str = ""
    model: str = ""
    full_url: str = ""
    remote_host: str = ""
    local_host: str = ""
    image_path: str = ""
    mjpeg_path: str = ""
    mjpeg_full_url: str = ""
    image_params: bool = False
    username: str = ""
    password: str = ""
    timeout_s: float = 10.0
    refresh_s: float = 0.0

    def snapshot_url(self, *, local: bool = False) -> str:
        """The snapshot URL for this camera, host picked by ``local``."""
        if self.full_url:
            return self.full_url
        host = self.local_host if local else self.remote_host
        if not host:
            host = self.remote_host or self.local_host
        return f"http://{host}{self.image_path}"

    def mjpeg_url(self, *, local: bool = False) -> str:
        """The MJPEG stream URL, or "" when the camera has none."""
        if self.mjpeg_full_url:
            return self.mjpeg_full_url
        if not self.mjpeg_path:
            return ""
        host = self.local_host if local else self.remote_host
        if not host:
            host = self.remote_host or self.local_host
        return f"http://{host}{self.mjpeg_path}"

    def as_dict(self) -> dict[str, Any]:
        """Back to the viewer JSON shape (password kept, never logged)."""
        data: dict[str, Any] = {
            "name": self.name,
            "location": self.location,
            "vendor": self.vendor,
            "type": self.model,
        }
        if self.full_url:
            data["full-url"] = self.full_url
        else:
            data["remote-host"] = self.remote_host
            data["local-host"] = self.local_host
            data["image"] = self.image_path
        if self.mjpeg_full_url:
            data["mjpeg-full-url"] = self.mjpeg_full_url
        elif self.mjpeg_path:
            data["mjpeg"] = self.mjpeg_path
        if self.image_params:
            data["image-params"] = True
        if self.username:
            data["username"] = self.username
        if self.password:
            data["password"] = self.password
        return data


def _str(value: Any) -> str:
    return "" if value is None else str(value)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def camera_from_dict(entry: Any, *, index: int = 0) -> CameraConfig:
    """One camera entry from the viewer JSON shape, validated."""
    if not isinstance(entry, dict):
        raise CamWallError(
            f"camera {index} is not an object — the cameras config "
            f"is a list of camera objects",
            kind="bad_config")
    full_url = _str(entry.get("full-url"))
    remote_host = _str(entry.get("remote-host"))
    local_host = _str(entry.get("local-host"))
    image_path = _str(entry.get("image"))
    if full_url and (remote_host or local_host or image_path):
        raise CamWallError(
            f"camera {index} has both a full-url and a host+path "
            f"form — pick one",
            kind="bad_url")
    if not full_url and not (remote_host or local_host) and not image_path:
        raise CamWallError(
            f"camera {index} has no URL — give it a full-url or a "
            f"remote-host/local-host plus an image path",
            kind="bad_url")
    if not full_url and (remote_host or local_host) and not image_path:
        raise CamWallError(
            f"camera {index} has a host but no image path",
            kind="bad_url")
    if full_url and not (full_url.startswith("http://")
                         or full_url.startswith("https://")):
        raise CamWallError(
            f"camera {index} full-url is not http(s): {full_url!r}",
            kind="bad_url")
    name = _str(entry.get("name")) or _str(entry.get("location")) \
        or f"camera-{index}"
    # The viewer joins host + image path verbatim, so a path
    # without its leading slash builds a broken URL
    # ("host" + "b.jpg" = "hostb.jpg"). The clone's example always
    # carries the slash; a config that omits it gets it added.
    if image_path and not image_path.startswith("/"):
        image_path = "/" + image_path
    mjpeg_path = _str(entry.get("mjpeg"))
    if mjpeg_path and not mjpeg_path.startswith("/"):
        mjpeg_path = "/" + mjpeg_path
    return CameraConfig(
        name=name,
        location=_str(entry.get("location")),
        vendor=_str(entry.get("vendor")),
        model=_str(entry.get("type")),
        full_url=full_url,
        remote_host=remote_host,
        local_host=local_host,
        image_path=image_path,
        mjpeg_path=mjpeg_path,
        mjpeg_full_url=_str(entry.get("mjpeg-full-url")),
        image_params=bool(entry.get("image-params")),
        username=_str(entry.get("username")),
        password=_str(entry.get("password")),
        timeout_s=_float(entry.get("timeout_s"), 10.0),
        refresh_s=_float(entry.get("refresh_s"), 0.0),
    )


def cameras_from_json(data: Any) -> list[CameraConfig]:
    """The camera list out of a parsed viewer JSON document."""
    if isinstance(data, dict):
        entries = data.get("cameras")
        if entries is None:
            raise CamWallError(
                "cameras config has no 'cameras' list",
                kind="bad_config")
    elif isinstance(data, list):
        entries = data
    else:
        raise CamWallError(
            "cameras config must be an object with a 'cameras' "
            "list, or a bare list of cameras",
            kind="bad_config")
    if not isinstance(entries, list):
        raise CamWallError(
            "cameras config 'cameras' is not a list",
            kind="bad_config")
    if not entries:
        raise CamWallError(
            "cameras config names no cameras",
            kind="no_cameras")
    return [camera_from_dict(entry, index=i)
            for i, entry in enumerate(entries)]


def load_config(source: Any) -> list[CameraConfig]:
    """Load cameras from a dict, a JSON string, a file, or a path.

    ``source`` may be a parsed document (dict/list), a JSON string,
    a :class:`pathlib.Path`, or a path string. A missing default
    path is a named refusal, never an empty wall.
    """
    if isinstance(source, (dict, list)):
        return cameras_from_json(source)
    if isinstance(source, Path):
        path = source
    elif isinstance(source, str) and source.strip().startswith(("{", "[")):
        try:
            return cameras_from_json(json.loads(source))
        except json.JSONDecodeError as exc:
            raise CamWallError(
                f"cameras config is not valid JSON: {exc}",
                kind="bad_config") from exc
    else:
        path = Path(str(source)).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CamWallError(
            f"cannot read cameras config {path}: {exc}",
            kind="bad_config") from exc
    try:
        return cameras_from_json(json.loads(text))
    except json.JSONDecodeError as exc:
        raise CamWallError(
            f"cameras config {path} is not valid JSON: {exc}",
            kind="bad_config") from exc


def dump_config(cameras: list[CameraConfig], path: str | Path) -> str:
    """Write cameras back to the viewer JSON shape."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {"cameras": [camera.as_dict() for camera in cameras]}
    target.write_text(json.dumps(document, indent=2) + "\n",
                      encoding="utf-8")
    return str(target)
