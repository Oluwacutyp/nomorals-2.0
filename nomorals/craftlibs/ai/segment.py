"""The ``image.segment`` contract (Craft-AI port of ai-core
``segment.rs``).

What every engine reads from a selection request and what every app
reads back, so a click, a box or "nothing there" means the same on
every route. A request carries the picture as its ``source`` input,
an optional description as its ``prompt`` input, and clicks or a box
in ``params``. Positions are fractions of the picture as a person
sees it (0 to 1, from the top left), so they survive any resizing on
the way. The answer is a grayscale PNG mask the size of the picture,
white where selected, then one ``application/json`` output,
:class:`Found`: whether anything matched, how sure the model is and,
when a model can tell, what the selection is. A route that finds
nothing still returns an empty mask and ``matched: false``, so "no
match" is never confused with a failed job.

Everything read here may come from another program, so it is
checked: positions inside the picture, a bounded number of clicks,
a short plain name. Outputs are the seam's job-event dicts
(``{"type": "output", "payload": ...}``), the shape
:class:`~nomorals.craftlibs.ai.job.JobSnapshot` returns.
"""

from __future__ import annotations

import json

from .error import AiError
from .model import InputRole
from .payload import Payload

__all__ = [
    "CAPABILITY",
    "FOUND_MEDIA_TYPE",
    "MASK_MEDIA_TYPE",
    "MAX_NAME_CHARS",
    "MAX_POINTS",
    "Area",
    "Click",
    "Found",
    "area",
    "clicks",
    "description",
    "found_output",
    "mask",
    "read_found",
    "with_area",
    "with_clicks",
]

#: The capability id this module describes.
CAPABILITY = "image.segment"
#: Most clicks one request may carry.
MAX_POINTS = 32
#: Longest name a result may give its selection, in characters.
MAX_NAME_CHARS = 64
#: The media type of the :class:`Found` output.
FOUND_MEDIA_TYPE = "application/json"
#: The media type of the mask: the contract says an inline PNG.
MASK_MEDIA_TYPE = "image/png"


class Click:
    """One click, as a fraction of the picture's width and height."""

    def __init__(self, x: float, y: float, positive: bool = True) -> None:
        self.x = float(x)
        self.y = float(y)
        self.positive = bool(positive)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "positive": self.positive}

    @classmethod
    def from_dict(cls, data: dict) -> Click:
        return cls(float(data["x"]), float(data["y"]),
                   bool(data.get("positive", True)))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Click)
                and self.to_dict() == other.to_dict())

    def __repr__(self) -> str:
        return f"Click({self.x}, {self.y}, {self.positive})"


class Area:
    """A box around the thing to select, as fractions of the picture:
    top left, then bottom right."""

    def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.x1 = float(x1)
        self.y1 = float(y1)

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_dict(cls, data: dict) -> Area:
        return cls(float(data["x0"]), float(data["y0"]),
                   float(data["x1"]), float(data["y1"]))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Area)
                and self.to_dict() == other.to_dict())

    def __repr__(self) -> str:
        return f"Area({self.x0}, {self.y0}, {self.x1}, {self.y1})"


class Found:
    """What a selection found, sent beside the mask."""

    def __init__(self, matched: bool = False, score=None,
                 name: str | None = None) -> None:
        self.matched = bool(matched)
        self.score = score
        self.name = name

    def to_dict(self) -> dict:
        data: dict = {"matched": self.matched}
        if self.score is not None:
            data["score"] = self.score
        if self.name is not None:
            data["name"] = self.name
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Found:
        return cls(bool(data.get("matched", False)),
                   data.get("score"), data.get("name"))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Found)
                and self.to_dict() == other.to_dict())

    def __repr__(self) -> str:
        return (f"Found(matched={self.matched}, score={self.score}, "
                f"name={self.name!r})")


def _fraction(value) -> bool:
    """Whether ``value`` is a usable position: a finite number from
    0 to 1."""
    return (isinstance(value, (int, float)) and value == value
            and value not in (float("inf"), float("-inf"))
            and 0.0 <= float(value) <= 1.0)


def with_clicks(request, clicks_list: list) -> object:
    """``request`` with ``clicks`` written as its ``points``
    parameter."""
    request.params["points"] = [c.to_dict() for c in clicks_list]
    return request


def with_area(request, area_value: Area) -> object:
    """``request`` with ``area`` written as its ``box`` parameter."""
    request.params["box"] = area_value.to_dict()
    return request


def clicks(request) -> list:
    """The clicks in ``request``, empty when it has none. Refused
    when ``points`` is not a list of at most :data:`MAX_POINTS`
    clicks inside the picture."""
    value = request.params.get("points")
    if value is None:
        return []
    wrong = AiError.invalid(
        "points must be a list of {x, y, positive} inside the picture")
    if not isinstance(value, list):
        raise wrong
    if len(value) > MAX_POINTS:
        raise AiError.invalid(f"at most {MAX_POINTS} points")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise wrong
        try:
            click = Click.from_dict(item)
        except (KeyError, TypeError, ValueError):
            raise wrong from None
        if _fraction(click.x) and _fraction(click.y):
            result.append(click)
        else:
            raise wrong
    return result


def area(request):
    """The box in ``request``, if any. Refused when it is not inside
    the picture with its top left above and left of its bottom
    right."""
    value = request.params.get("box")
    if value is None:
        return None
    wrong = AiError.invalid(
        "box must be {x0, y0, x1, y1} inside the picture, top left first")
    if not isinstance(value, dict):
        raise wrong
    try:
        box = Area.from_dict(value)
    except (KeyError, TypeError, ValueError):
        raise wrong from None
    inside = all(_fraction(v) for v in
                 (box.x0, box.y0, box.x1, box.y1))
    if inside and box.x0 < box.x1 and box.y0 < box.y1:
        return box
    raise wrong


def description(request):
    """The description in ``request``'s prompt input, if it has one
    that is not blank."""
    for role, payload in request.inputs:
        if (role == InputRole.PROMPT and payload.as_text() is not None
                and payload.as_text().strip()):
            return payload.as_text()
    return None


def found_output(found: Found) -> dict:
    """``found`` as the output event that goes beside the mask."""
    try:
        data = json.dumps(found.to_dict()).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AiError.invalid(f"selection result: {exc}") from exc
    return {"type": "output",
            "payload": Payload.inline(FOUND_MEDIA_TYPE, data).to_dict()}


def read_found(outputs: list):
    """The first output that is a valid :class:`Found`, if any. A
    name that is too long or holds control characters is dropped,
    and a score outside 0 to 1 is dropped, so neither can reach the
    screen."""
    for event in outputs:
        if not isinstance(event, dict) or event.get("type") != "output":
            continue
        payload = event.get("payload") or {}
        if payload.get("type") != "inline":
            continue
        if payload.get("media_type") != FOUND_MEDIA_TYPE:
            continue
        try:
            raw = Payload.from_dict(payload).data or b""
            found = Found.from_dict(json.loads(raw.decode("utf-8")))
        except (ValueError, TypeError, KeyError):
            continue
        if found.score is not None and not _fraction(found.score):
            found.score = None
        if found.name is not None:
            name = found.name
            if (not name.strip() or len(name) > MAX_NAME_CHARS
                    or any(ord(c) < 0x20 or ord(c) == 0x7F for c in name)):
                found.name = None
        return found
    return None


def mask(outputs: list):
    """The first inline PNG output: the mask. A preview in another
    format, or an image that is only a handle, is not the mask and
    is skipped."""
    for event in outputs:
        if not isinstance(event, dict) or event.get("type") != "output":
            continue
        payload = event.get("payload") or {}
        if (payload.get("type") == "inline"
                and payload.get("media_type") == MASK_MEDIA_TYPE):
            return payload
    return None
