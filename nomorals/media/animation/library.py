"""Animation library — named presets, each a real timeline.

Every preset is built from the same model the editor uses (layers,
keyframes, tweens, motion paths), so a preset can be opened, edited,
rendered and published exactly like a document the owner made by
hand. Presets are searchable by name.
"""

from __future__ import annotations

from .easing import Easing
from .model import AnimationDocument, Element, Transform

__all__ = ["build_preset", "list_presets", "search_presets"]

_PRESETS: dict[str, str] = {
    "walk-cycle": "A character walks in place: legs swing, body bobs.",
    "bounce": "A ball bounces with squash and stretch.",
    "fade-in": "A title fades in from transparent.",
    "slide-in": "A card slides in from off-stage left.",
    "pulse": "A button pulses, growing and shrinking in a loop.",
    "typewriter": "Text appears one character at a time.",
}


def list_presets() -> list[str]:
    return sorted(_PRESETS)


def search_presets(query: str) -> list[str]:
    q = query.strip().lower()
    return [name for name in list_presets()
            if q in name or q in _PRESETS[name].lower()]


def build_preset(name: str) -> AnimationDocument:
    from .errors import AnimationError

    builders = {
        "walk-cycle": _walk_cycle,
        "bounce": _bounce,
        "fade-in": _fade_in,
        "slide-in": _slide_in,
        "pulse": _pulse,
        "typewriter": _typewriter,
    }
    builder = builders.get(name)
    if builder is None:
        raise AnimationError(
            f"unknown preset {name!r}; pick from {list_presets()}")
    return builder()


def _base(name: str, fps: float = 24.0) -> AnimationDocument:
    return AnimationDocument.create(name, width=320, height=180, fps=fps)


def _walk_cycle() -> AnimationDocument:
    """Two 8-frame cycles: two legs swing opposite, the body bobs."""
    doc = _base("walk-cycle")
    tl = doc.scenes[0].timeline
    body = tl.add_layer("body")
    first_torso: Element | None = None
    for i, start in enumerate((0, 4, 8, 12)):
        kf = body.add_keyframe(start, 4)
        torso = Element.ellipse(160.0, 60.0, 25.0, 30.0, fill="#4472c4")
        if first_torso is None:
            first_torso = torso
        else:
            torso.id = first_torso.id
        torso.transform = Transform(y=4.0 if i % 2 else 0.0)
        kf.add_element(torso)
        if i > 0:
            body.keyframes[i - 1].tween_to(
                kf, easing=Easing.named("sine-in-out"))
    legs = tl.add_layer("legs")
    first_legs: list[Element] = []
    for i, start in enumerate((0, 4, 8, 12)):
        swing = -30.0 if i % 2 == 0 else 30.0
        kf = legs.add_keyframe(start, 4)
        leg = Element.rect(150.0, 85.0, 8.0, 45.0, fill="#222222")
        leg.transform = Transform(rotation=swing)
        kf.add_element(leg)
        leg2 = Element.rect(165.0, 85.0, 8.0, 45.0, fill="#222222")
        leg2.transform = Transform(rotation=-swing)
        kf.add_element(leg2)
        if i == 0:
            first_legs = [leg, leg2]
        else:
            # Same ids as the first keyframe, so the tween matches them.
            leg.id, leg2.id = first_legs[0].id, first_legs[1].id
            legs.keyframes[i - 1].tween_to(
                kf, easing=Easing.named("sine-in-out"))
    return doc


def _bounce() -> AnimationDocument:
    """24-frame bounce: fall, squash at the bottom, rise again."""
    doc = _base("bounce")
    tl = doc.scenes[0].timeline
    layer = tl.add_layer("ball")
    kf0 = layer.add_keyframe(0, 12)
    ball = Element.ellipse(160.0, 20.0, 20.0, 20.0, fill="#e74c3c")
    kf0.add_element(ball)
    kf1 = layer.add_keyframe(12, 12)
    ball2 = Element.ellipse(160.0, 20.0, 20.0, 20.0, fill="#e74c3c")
    ball2.id = ball.id
    ball2.transform = Transform(y=120.0, scale_x=1.25, scale_y=0.75)
    kf1.add_element(ball2)
    kf0.tween_to(kf1, easing=Easing.named("bounce-out"),
                 keep_area=True)
    kf2 = layer.add_keyframe(24, 12)
    ball3 = Element.ellipse(160.0, 20.0, 20.0, 20.0, fill="#e74c3c")
    ball3.id = ball.id
    ball3.transform = Transform(y=0.0, scale_x=1.0, scale_y=1.0)
    kf2.add_element(ball3)
    kf1.tween_to(kf2, easing=Easing.named("quad-out"),
                 keep_area=True)
    return doc


def _fade_in() -> AnimationDocument:
    doc = _base("fade-in")
    tl = doc.scenes[0].timeline
    layer = tl.add_layer("title")
    kf0 = layer.add_keyframe(0, 12)
    title = Element.text("Hello", 60.0, 70.0, font_size=48.0,
                         color="#222222")
    title.transform = Transform(opacity=0.0)
    kf0.add_element(title)
    kf1 = layer.add_keyframe(12, 12)
    title2 = Element.text("Hello", 60.0, 70.0, font_size=48.0,
                          color="#222222")
    title2.id = title.id
    title2.transform = Transform(opacity=1.0)
    kf1.add_element(title2)
    kf0.tween_to(kf1, easing=Easing.named("sine-in-out"))
    return doc


def _slide_in() -> AnimationDocument:
    doc = _base("slide-in")
    tl = doc.scenes[0].timeline
    layer = tl.add_layer("card")
    kf0 = layer.add_keyframe(0, 12)
    card = Element.rect(0.0, 60.0, 140.0, 60.0, fill="#2ecc71")
    card.transform = Transform(x=-160.0)
    kf0.add_element(card)
    kf1 = layer.add_keyframe(12, 12)
    card2 = Element.rect(0.0, 60.0, 140.0, 60.0, fill="#2ecc71")
    card2.id = card.id
    card2.transform = Transform(x=90.0)
    kf1.add_element(card2)
    kf0.tween_to(kf1, easing=Easing.named("quad-out"))
    return doc


def _pulse() -> AnimationDocument:
    doc = _base("pulse")
    tl = doc.scenes[0].timeline
    layer = tl.add_layer("button")
    kf0 = layer.add_keyframe(0, 6)
    btn = Element.ellipse(160.0, 90.0, 40.0, 25.0, fill="#9b59b6")
    kf0.add_element(btn)
    kf1 = layer.add_keyframe(6, 6)
    btn2 = Element.ellipse(160.0, 90.0, 40.0, 25.0, fill="#9b59b6")
    btn2.id = btn.id
    btn2.transform = Transform(scale_x=1.2, scale_y=1.2)
    kf1.add_element(btn2)
    kf0.tween_to(kf1, easing=Easing.named("sine-in-out"))
    kf2 = layer.add_keyframe(12, 6)
    btn3 = Element.ellipse(160.0, 90.0, 40.0, 25.0, fill="#9b59b6")
    btn3.id = btn.id
    kf2.add_element(btn3)
    kf1.tween_to(kf2, easing=Easing.named("sine-in-out"))
    return doc


def _typewriter() -> AnimationDocument:
    doc = _base("typewriter")
    tl = doc.scenes[0].timeline
    layer = tl.add_layer("text")
    word = "Hello!"
    for i, _ in enumerate(word):
        kf = layer.add_keyframe(i * 2, 2)
        text = Element.text(word[:i + 1], 60.0, 70.0, font_size=40.0,
                            color="#222222")
        kf.add_element(text)
    hold = layer.add_keyframe(len(word) * 2, 12)
    full = Element.text(word, 60.0, 70.0, font_size=40.0,
                        color="#222222")
    hold.add_element(full)
    return doc
