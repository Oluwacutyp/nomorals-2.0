"""Phase 4: mock-screen animation — screen recordings
(census §9.2, §16.3).

FakeDetail's flagship behaviour is not the still: the
supported tools record an animated video of the preview.
A recording here is a DATA TIMELINE, not a screen
capture — the screen's own renderer produces the frame
states and Motion Studio's frame pipe carries them to
ffmpeg (the census's preferred foundation, §2.4):

- chats: messages reveal one by one; typing dots precede
  the contact's replies; a voice note plays for its TRUE
  decoded duration with its playhead advancing over the
  measured waveform;
- lists: rows arrive one by one at the top;
- WhatsApp Status / Instagram Story: segments advance one
  by one while each segment's bar fills for its set
  duration — the recording ends on exactly the previewed
  screen (its declared index / played fraction), then
  holds;
- calls: an ongoing call's timer advances; an incoming
  call holds ringing;
- profiles / comment sheets / video pages: the page
  scrolls through, chrome pinned, pausing at the top and
  the bottom;
- screens with no choreography of their own (a post, the
  views sheet, call history) record as a hold on the
  screen.

The video is silent (census §5.9: recorded video has no
sound) — the voice note's playback is the playhead
advancing for its measured duration, never an invented
waveform and never a guessed duration: undecodable audio
fails the recording with the reason, the media-in-screens
rule.

A timeline is data: :class:`TimelineSpec` carries every
duration (per-message reveal hold, typing length, per-
segment length, scroll time), so a recording's length is
a property of the screen and the spec, never of sleeps
in the renderer.  Frames are rendered lazily, one
distinct state at a time — a long chat renders each
prefix once, not once per video frame.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from PIL import Image, ImageDraw

from .designs import CHAT_DESIGNS, DEVICES
from .media import MediaError, probe_audio
from .render import MockScreensError, render_screen
from .render_screens import (
    render_call_screen,
    render_list_screen,
    render_status_screen,
)
from .render_social import (
    render_comment_screen,
    render_profile_screen,
    render_story_screen,
    render_video_screen,
)
from .schema import (
    CallScreen,
    CommentScreen,
    ListScreen,
    ProfileScreen,
    Screen,
    StatusScreen,
    StoryScreen,
    VideoScreen,
)


class RecordingError(Exception):
    """A recording that cannot be made; the message IS the
    user-fit reason (bad media, no ffmpeg, a probe that
    disagrees with the encode)."""


# ── the timeline spec (data, not sleeps) ─────────────────


@dataclass
class TimelineSpec:
    """Every duration a recording is built from.  The
    defaults are the organ's house timing; a caller may
    name faster or slower values, but a duration is never
    invented per screen — it comes from this spec or from
    measured media (a voice note's true length, a
    segment's set length)."""

    fps: float = 24.0
    start_hold_s: float = 0.5
    reveal_hold_s: float = 0.7
    typing_s: float = 1.2
    typing_phase_s: float = 0.3
    voice_tail_s: float = 0.3
    voice_step_s: float = 0.25
    segment_s: float = 5.0
    progress_step_s: float = 0.25
    scroll_s: float = 4.0
    scroll_pause_s: float = 0.6
    call_hold_s: float = 4.0
    hold_s: float = 4.0
    end_hold_s: float = 1.0

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise RecordingError(
                "a recording needs a positive fps, got "
                + repr(self.fps))
        for name in ("start_hold_s", "reveal_hold_s",
                     "typing_s", "typing_phase_s",
                     "voice_tail_s", "voice_step_s",
                     "segment_s", "progress_step_s",
                     "scroll_s", "scroll_pause_s",
                     "call_hold_s", "hold_s", "end_hold_s"):
            if float(getattr(self, name)) < 0:
                raise RecordingError(
                    "timeline durations cannot be negative: "
                    + name + " = "
                    + repr(getattr(self, name)))


# ── clips + recordings ───────────────────────────────────


@dataclass
class Clip:
    """One distinct frame state and how long it shows.
    ``render`` is called at most once per recording pass
    (the recorder caches by clip) — a 24 fps video of a
    9-message chat renders ~25 frames, not 200+."""

    seconds: float
    label: str
    render: Callable[[], Image.Image]


@dataclass
class Recording:
    """A built recording: its clips, its fixed frame size,
    and the final still's drawn record (the video ends on
    exactly that screen)."""

    kind: str
    spec: TimelineSpec
    clips: list = field(default_factory=list)
    size: tuple = (0, 0)
    drawn: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._cache: dict = {}

    @property
    def duration_s(self) -> float:
        """The timeline's own length — the exported
        video's duration is checked against this."""
        return sum(float(c.seconds) for c in self.clips)

    @property
    def timeline(self) -> list:
        """The clip plan as data: label + start/end
        seconds, for the record and for tests."""
        out: list = []
        at = 0.0
        for clip in self.clips:
            out.append({"label": clip.label,
                        "start_s": round(at, 3),
                        "end_s": round(at + float(clip.seconds), 3)})
            at += float(clip.seconds)
        return out

    def frame_at(self, t: float) -> Image.Image:
        """The frame at second ``t`` — the clip covering
        it, rendered once and cached by clip index."""
        at = 0.0
        for idx, clip in enumerate(self.clips):
            end = at + float(clip.seconds)
            if t < end or idx == len(self.clips) - 1:
                if idx not in self._cache:
                    self._cache[idx] = clip.render()
                return self._cache[idx]
            at = end
        return self.clips[-1].render()


def _png_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _render_png(fn, *args, **kw):
    data, drawn = fn(*args, **kw)
    return _png_image(data), drawn


def _clip(seconds: float, label: str, render) -> Clip:
    return Clip(seconds=max(0.0, float(seconds)),
                label=label, render=render)


# ── fixed-canvas compositing ─────────────────────────────
#
# Chat and list renders are content-height: a prefix is
# shorter than the finished screen.  A video frame is
# one fixed size, so each prefix is composited into the
# finished size — chrome pinned top and bottom, the
# content anchored the way the app anchors it (chat
# content grows upward from the input bar; list rows
# arrive at the top).


def _scale_for(drawn: dict, style: str) -> float:
    width = int(DEVICES[style]["width"])
    return float(drawn["size"][0]) / float(width)


def _composite_chat(prefix: Image.Image, drawn: dict,
                    size: tuple, wallpaper) -> Image.Image:
    """A chat prefix into the finished frame: status bar +
    header pinned at the top, input bar + nav pinned at
    the bottom, the messages bottom-aligned between."""
    width, height = size
    regions = drawn["regions"]
    header_bottom = int(regions["header"][1]
                        + regions["header"][3])
    input_top = int(regions["input_bar"][1])
    frame = Image.new("RGB", size, wallpaper)
    frame.paste(prefix.crop((0, 0, width, header_bottom)),
                (0, 0))
    bottom = prefix.crop((0, input_top, width, prefix.height))
    frame.paste(bottom, (0, height - bottom.height))
    content = prefix.crop((0, header_bottom, width, input_top))
    frame.paste(content,
                (0, height - bottom.height - content.height))
    return frame


def _composite_list(prefix: Image.Image, drawn: dict,
                    size: tuple, bg) -> Image.Image:
    """A list prefix into the finished frame: chrome
    pinned at the top, tab bar pinned at the bottom, the
    rows top-aligned under the header (newest first)."""
    width, height = size
    regions = drawn["regions"]
    top_h = 0
    for name in ("status_bar", "header", "search",
                 "stories", "tabs"):
        region = regions.get(name)
        if region:
            top_h = max(top_h, int(region[1] + region[3]))
    bottom_top = prefix.height
    for name in ("tab_bar", "nav"):
        region = regions.get(name)
        if region:
            bottom_top = min(bottom_top, int(region[1]))
    frame = Image.new("RGB", size, bg)
    frame.paste(prefix.crop((0, 0, width, top_h)), (0, 0))
    bottom = prefix.crop((0, bottom_top, width, prefix.height))
    frame.paste(bottom, (0, height - bottom.height))
    content = prefix.crop((0, top_h, width, bottom_top))
    frame.paste(content, (0, top_h))
    return frame


def _scroll_frame(full: Image.Image, drawn: dict, size: tuple,
                  offset: int, *, pin_top: int,
                  pin_bottom: int) -> Image.Image:
    """One scroll position of a long page: the pinned top
    and bottom strips stay, the window between them slides
    through the page."""
    width, height = size
    frame = Image.new("RGB", size, (0, 0, 0))
    frame.paste(full.crop((0, 0, width, pin_top)), (0, 0))
    if pin_bottom:
        bottom = full.crop((0, full.height - pin_bottom,
                            width, full.height))
        frame.paste(bottom, (0, height - pin_bottom))
    window_h = height - pin_top - pin_bottom
    window = full.crop((0, pin_top + offset, width,
                        pin_top + offset + window_h))
    frame.paste(window, (0, pin_top))
    return frame


# ── typing dots (an overlay, not a renderer change) ──────
#
# The chat renderer draws messages; the typing bubble is
# a transient overlay the animation module draws onto the
# composited frame, in the incoming bubble's own colour.


def _typing_frame(base: Image.Image, screen: Screen,
                  drawn: dict, phase: int,
                  wallpaper, size: tuple) -> Image.Image:
    design = CHAT_DESIGNS[screen.platform]
    scale = _scale_for(drawn, screen.chrome.style)
    regions = drawn["regions"]
    # the base frame is the composited final-size frame:
    # the prefix's content sits shifted down by the
    # height the prefix has not grown yet, so every
    # prefix coordinate moves by that shift.
    shift = int(size[1]) - int(drawn["size"][1])
    input_top = int(regions["input_bar"][1]) + shift
    # the incoming bubble colour, sampled from the last
    # incoming message on the frame (or the wallpaper's
    # tone when no message has arrived yet)
    bubble = None
    for entry in reversed(drawn.get("elements") or []):
        if entry.get("kind") in ("text", "image", "voice") \
                and entry.get("sender") not in ("me", "system"):
            box = entry.get("box") or []
            if len(box) == 4 and box[2] > 10 and box[3] > 10:
                px = base.getpixel(
                    (int(box[0] + box[2] // 2),
                     int(box[1] + 2) + shift))
                bubble = tuple(px[:3])
            break
    if bubble is None:
        lum = sum(wallpaper[:3]) / 3.0
        bubble = (235, 235, 235) if lum > 128 else (52, 52, 52)
    frame = base.copy()
    draw = ImageDraw.Draw(frame)
    avatar_w = 0.0
    if design.get("run_avatar"):
        group_only = bool(
            design.get("run_avatar_group_only", False))
        if not group_only or screen.group:
            avatar_w = float(design.get("run_avatar_size", 24))
    bx = int(round((16 + avatar_w) * scale))
    bw = int(round(56 * scale))
    bh = int(round(30 * scale))
    by = input_top - int(round(6 * scale)) - bh
    draw.rounded_rectangle(
        [bx, by, bx + bw, by + bh],
        radius=int(round(9 * scale)), fill=bubble)
    lum = sum(bubble) / 3.0
    dot = (64, 64, 64) if lum > 128 else (215, 215, 215)
    idle = tuple((d + b) // 2 for d, b in zip(
        dot, bubble, strict=True))
    for i in range(3):
        cx = bx + int(round((15 + i * 11) * scale))
        cy = by + bh // 2 - (int(round(2 * scale))
                             if i == phase % 3 else 0)
        r = int(round(3 * scale))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=dot if i == phase % 3 else idle)
    return frame


# ── voice probing (the timeline needs true lengths) ──────


def _voice_seconds(msg, media: dict | None) -> float | None:
    """A voice message's TRUE duration, measured from its
    audio — or None when it carries no media (the
    no-media case plays nothing).  Undecodable audio
    raises RecordingError with the reason: no waveform
    is ever invented, and no duration is ever guessed."""
    source = None
    if media:
        source = media.get(msg.id)
    if source is None:
        source = getattr(msg, "media", "") or ""
    if not source:
        return None
    try:
        return float(probe_audio(source).seconds)
    except MediaError as exc:
        raise RecordingError(
            "the voice note on message " + repr(msg.id)
            + " could not be used: " + str(exc)) from exc


def _render_chat(screen: Screen, *, avatars, media,
                 voice_progress=None):
    return _render_png(
        render_screen, screen, avatars=avatars, media=media,
        voice_progress=voice_progress or {})


# ── chats ────────────────────────────────────────────────


def build_chat_recording(screen: Screen, spec=None, *,
                         avatars: dict | None = None,
                         media: dict | None = None) -> Recording:
    """The flagship recording (census §9.2): messages
    reveal one by one, typing dots precede each incoming
    run, a real voice note plays for its measured
    duration with the playhead advancing, and the video
    ends on exactly the previewed screen."""
    spec = spec or TimelineSpec()
    try:
        _full, full_drawn = _render_chat(
            screen, avatars=avatars, media=media)
    except MockScreensError as exc:
        raise RecordingError(str(exc)) from exc
    size = tuple(full_drawn["size"])
    regions = full_drawn["regions"]
    header_bottom = int(regions["header"][1]
                        + regions["header"][3])
    wallpaper = _full.getpixel(
        (size[0] // 2, min(header_bottom + 2, size[1] - 1)))
    # the voice lengths, measured once up front (a bad
    # file fails the recording before any frame renders)
    voice_len: dict = {}
    for msg in screen.messages:
        if msg.kind == "voice":
            voice_len[msg.id] = _voice_seconds(msg, media)
    clips: list = []

    def _prefix_parts(i: int, voice_progress=None):
        part = replace(screen, messages=list(
            screen.messages[:i]))
        img, drawn = _render_chat(
            part, avatars=avatars, media=media,
            voice_progress=voice_progress)
        return _composite_chat(img, drawn, size, wallpaper), drawn

    def _prefix(i: int, voice_progress=None):
        return _prefix_parts(i, voice_progress)[0]

    def _typing(i: int, phase: int):
        frame, drawn = _prefix_parts(i)
        return _typing_frame(frame, screen, drawn, phase,
                             wallpaper, size)

    clips.append(_clip(
        spec.start_hold_s, "opening",
        lambda: _prefix(0)))
    messages = screen.messages
    for i, msg in enumerate(messages):
        incoming = msg.sender not in ("me", "system") \
            and msg.kind not in ("divider", "system")
        run_start = incoming and (
            i == 0 or messages[i - 1].sender != msg.sender
            or messages[i - 1].kind in ("divider", "system"))
        if run_start and spec.typing_s > 0:
            phases = max(1, int(math.ceil(
                spec.typing_s / spec.typing_phase_s)))
            for k in range(phases):
                seconds = min(
                    spec.typing_phase_s,
                    spec.typing_s - k * spec.typing_phase_s)
                clips.append(_clip(
                    seconds, "typing",
                    lambda i=i, k=k: _typing(i, k % 3)))
        length = voice_len.get(msg.id)
        if msg.kind == "voice" and length:
            clips.append(_clip(
                spec.reveal_hold_s, "reveal " + msg.id,
                lambda i=i, msg=msg: _prefix(
                    i + 1, voice_progress={msg.id: 0.0})))
            steps = max(1, int(math.ceil(
                length / spec.voice_step_s)))
            for k in range(1, steps + 1):
                frac = min(1.0, k * spec.voice_step_s / length)
                seconds = min(
                    spec.voice_step_s,
                    length - (k - 1) * spec.voice_step_s)
                clips.append(_clip(
                    seconds, "voice " + msg.id,
                    lambda i=i, msg=msg, frac=frac: _prefix(
                        i + 1,
                        voice_progress={msg.id: frac})))
            clips.append(_clip(
                spec.voice_tail_s, "voice tail " + msg.id,
                lambda i=i, msg=msg: _prefix(
                    i + 1, voice_progress={msg.id: 1.0})))
        else:
            clips.append(_clip(
                spec.reveal_hold_s, "reveal " + msg.id,
                lambda i=i: _prefix(i + 1)))
    clips.append(_clip(
        spec.end_hold_s, "ending", lambda: _prefix(len(messages))))
    return Recording(kind="chat", spec=spec, clips=clips,
                     size=size, drawn=full_drawn)


# ── lists ────────────────────────────────────────────────


def build_list_recording(screen: ListScreen, spec=None, *,
                         avatars: dict | None = None,
                         media: dict | None = None) -> Recording:
    """Rows arrive one by one at the top (census §9.2) —
    the list recording.  (The Telegram list's swipe-by-
    swipe scroll with pauses is a distinct choreography
    the census names without a frame model; row arrival
    is the shared list behaviour shipped here.)"""
    spec = spec or TimelineSpec()
    try:
        _full, full_drawn = _render_png(
            render_list_screen, screen, avatars=avatars,
            media=media)
    except MockScreensError as exc:
        raise RecordingError(str(exc)) from exc
    size = tuple(full_drawn["size"])
    empty_img, empty_drawn = _render_png(
        render_list_screen, replace(screen, rows=[]),
        avatars=avatars, media=media)
    regions = empty_drawn["regions"]
    top_h = 0
    for name in ("status_bar", "header", "search",
                 "stories", "tabs"):
        region = regions.get(name)
        if region:
            top_h = max(top_h, int(region[1] + region[3]))
    bg = empty_img.getpixel(
        (size[0] // 2, min(top_h + 2, size[1] - 1)))
    clips: list = []

    def _prefix(i: int):
        part = replace(screen, rows=list(screen.rows[:i]))
        img, drawn = _render_png(
            render_list_screen, part, avatars=avatars,
            media=media)
        return _composite_list(img, drawn, size, bg)

    clips.append(_clip(
        spec.start_hold_s, "opening",
        lambda: _prefix(0)))
    for i, row in enumerate(screen.rows):
        clips.append(_clip(
            spec.reveal_hold_s, "reveal " + row.id,
            lambda i=i: _prefix(i + 1)))
    clips.append(_clip(
        spec.end_hold_s, "ending",
        lambda: _prefix(len(screen.rows))))
    return Recording(kind="list", spec=spec, clips=clips,
                     size=size, drawn=full_drawn)


# ── status / story (the segmented progress model) ────────


def _progress_recording(kind: str, screen, render_fn, spec,
                        *, avatars, media) -> Recording:
    """Segments advance one by one while each segment's
    bar fills for its set duration (census §9.2) — and
    the recording ends on exactly the previewed screen:
    its declared index, filled to its declared played
    fraction, then held."""
    try:
        _full, full_drawn = _render_png(
            render_fn, screen, avatars=avatars, media=media)
    except MockScreensError as exc:
        raise RecordingError(str(exc)) from exc
    size = tuple(full_drawn["size"])
    clips: list = []

    def _at(index: int, played: float):
        part = replace(screen, index=index, played=played)
        img, _drawn = _render_png(
            render_fn, part, avatars=avatars, media=media)
        return img

    clips.append(_clip(
        spec.start_hold_s, "opening",
        lambda: _at(0, 0.0)))
    for seg in range(int(screen.index) + 1):
        target = float(screen.played) if seg == screen.index \
            else 1.0
        if target <= 0:
            continue
        total = spec.segment_s * target
        steps = max(1, int(math.ceil(
            total / spec.progress_step_s)))
        for k in range(1, steps + 1):
            frac = target * k / steps
            seconds = total / steps
            clips.append(_clip(
                seconds, "segment " + str(seg),
                lambda seg=seg, frac=frac: _at(seg, frac)))
    clips.append(_clip(
        spec.end_hold_s, "ending",
        lambda: _at(int(screen.index),
                    float(screen.played))))
    return Recording(kind=kind, spec=spec, clips=clips,
                     size=size, drawn=full_drawn)


def build_status_recording(screen: StatusScreen, spec=None, *,
                           avatars: dict | None = None,
                           media: dict | None = None
                           ) -> Recording:
    """A WhatsApp Status recording: the current progress
    segment fills to the user-set point (census §9.2)."""
    spec = spec or TimelineSpec()
    return _progress_recording(
        "status", screen, render_status_screen, spec,
        avatars=avatars, media=media)


def build_story_recording(screen: StoryScreen, spec=None, *,
                          avatars: dict | None = None,
                          media: dict | None = None
                          ) -> Recording:
    """An Instagram Story recording: segments advance one
    by one while each segment's bar fills for its set
    duration (census §9.2)."""
    spec = spec or TimelineSpec()
    return _progress_recording(
        "story", screen, render_story_screen, spec,
        avatars=avatars, media=media)


# ── calls ────────────────────────────────────────────────

_TIMER_RE = re.compile(r"^(\d+):(\d{2})(?::(\d{2}))?$")


def _timer_after(status: str, seconds: float) -> str | None:
    """An ongoing call's timer text advanced by
    ``seconds`` — or None when the status line is not a
    timer (ringing text, a label)."""
    match = _TIMER_RE.match(str(status or "").strip())
    if match is None:
        return None
    if match.group(3) is not None:
        total = (int(match.group(1)) * 3600
                 + int(match.group(2)) * 60
                 + int(match.group(3)) + int(seconds))
        return (str(total // 3600) + ":"
                + format((total // 60) % 60, "02d") + ":"
                + format(total % 60, "02d"))
    total = (int(match.group(1)) * 60 + int(match.group(2))
             + int(seconds))
    return str(total // 60) + ":" + format(total % 60, "02d")


def build_call_recording(screen: CallScreen, spec=None, *,
                         avatars: dict | None = None,
                         media: dict | None = None) -> Recording:
    """A call recording: an ongoing call's timer advances
    second by second; an incoming call holds ringing.
    The video ends on the screen as authored."""
    spec = spec or TimelineSpec()
    try:
        _full, full_drawn = _render_png(
            render_call_screen, screen, avatars=avatars,
            media=media)
    except MockScreensError as exc:
        raise RecordingError(str(exc)) from exc
    size = tuple(full_drawn["size"])
    clips: list = []

    def _at(status: str):
        part = replace(screen, status=status)
        img, _drawn = _render_png(
            render_call_screen, part, avatars=avatars,
            media=media)
        return img

    clips.append(_clip(
        spec.start_hold_s, "opening",
        lambda: _at(screen.status)))
    whole = _timer_after(screen.status, spec.call_hold_s)
    if whole is None:
        clips.append(_clip(
            spec.call_hold_s, "call",
            lambda: _at(screen.status)))
    else:
        steps = max(1, int(math.ceil(spec.call_hold_s)))
        for k in range(1, steps + 1):
            seconds = min(1.0, spec.call_hold_s - (k - 1))
            elapsed = min(float(k), spec.call_hold_s)
            clips.append(_clip(
                seconds, "call",
                lambda elapsed=elapsed: _at(
                    _timer_after(screen.status, elapsed)
                    or screen.status)))
    clips.append(_clip(
        spec.end_hold_s, "ending",
        lambda: _at(whole if whole is not None
                    else screen.status)))
    return Recording(kind="call", spec=spec, clips=clips,
                     size=size, drawn=full_drawn)


# ── scrolling pages ──────────────────────────────────────


def build_scroll_recording(screen, render_fn, spec=None, *,
                           avatars: dict | None = None,
                           media: dict | None = None,
                           pin_top: int | None = None
                           ) -> Recording:
    """A long page recorded scrolling through it (census
    §9.2): the phone-height viewport slides down the page
    with the status bar pinned, pausing at the top and the
    bottom.  A page that fits one phone screen records as
    a hold — there is nothing to scroll.  ``pin_top``
    names extra pinned height above the scrolling window
    (the comment sheet's count header); the default pins
    only the status bar."""
    spec = spec or TimelineSpec()
    try:
        full, full_drawn = _render_png(
            render_fn, screen, avatars=avatars, media=media)
    except MockScreensError as exc:
        raise RecordingError(str(exc)) from exc
    size_full = full.size
    style = screen.chrome.style
    scale = float(size_full[0]) / float(DEVICES[style]["width"])
    viewport_h = int(round(
        float(DEVICES[style]["screen_h"]) * scale))
    regions = full_drawn["regions"]
    status = regions.get("status_bar") or [0, 0, 0, 0]
    top = int(pin_top) if pin_top is not None \
        else int(status[1] + status[3])
    nav = regions.get("nav")
    bottom = int(nav[3]) if nav else 0
    size = (size_full[0], viewport_h)
    clips: list = []

    def _at(offset: int):
        return _scroll_frame(
            full, full_drawn, size, offset,
            pin_top=top, pin_bottom=bottom)

    if size_full[1] <= viewport_h:
        clips.append(_clip(
            spec.start_hold_s + spec.end_hold_s, "page",
            lambda: full))
        return Recording(kind="scroll", spec=spec, clips=clips,
                         size=size_full, drawn=full_drawn)
    max_offset = size_full[1] - viewport_h
    clips.append(_clip(
        spec.start_hold_s + spec.scroll_pause_s, "top",
        lambda: _at(0)))
    steps = max(2, int(math.ceil(
        spec.scroll_s / spec.progress_step_s)))
    for k in range(1, steps + 1):
        offset = int(round(max_offset * k / steps))
        clips.append(_clip(
            spec.scroll_s / steps, "scroll",
            lambda offset=offset: _at(offset)))
    clips.append(_clip(
        spec.scroll_pause_s + spec.end_hold_s, "bottom",
        lambda: _at(max_offset)))
    return Recording(kind="scroll", spec=spec, clips=clips,
                     size=size, drawn=full_drawn)


def build_profile_recording(screen: ProfileScreen, spec=None,
                            *, avatars: dict | None = None,
                            media: dict | None = None
                            ) -> Recording:
    """A profile page recorded scrolling through it."""
    spec = spec or TimelineSpec()
    return build_scroll_recording(
        screen, render_profile_screen, spec,
        avatars=avatars, media=media)


def build_comment_recording(screen: CommentScreen, spec=None,
                            *, avatars: dict | None = None,
                            media: dict | None = None
                            ) -> Recording:
    """A comment sheet recorded scrolling through it —
    the sheet's count header stays pinned while the
    comments slide."""
    spec = spec or TimelineSpec()
    try:
        _full, full_drawn = _render_png(
            render_comment_screen, screen, avatars=avatars,
            media=media)
    except MockScreensError as exc:
        raise RecordingError(str(exc)) from exc
    pin_top = None
    for entry in full_drawn.get("elements") or []:
        box = entry.get("box") or []
        if len(box) == 4:
            pin_top = int(box[1])
            break
    return build_scroll_recording(
        screen, render_comment_screen, spec,
        avatars=avatars, media=media, pin_top=pin_top)


def build_video_recording(screen: VideoScreen, spec=None, *,
                          avatars: dict | None = None,
                          media: dict | None = None
                          ) -> Recording:
    """A YouTube video page recorded scrolling through
    it (census §5.9: download as a picture or record
    scrolling as video)."""
    spec = spec or TimelineSpec()
    return build_scroll_recording(
        screen, render_video_screen, spec,
        avatars=avatars, media=media)


# ── holds (screens with no choreography of their own) ───


def build_hold_recording(frame: Image.Image, spec=None, *,
                         kind: str = "hold",
                         drawn: dict | None = None) -> Recording:
    """A screen with no animation of its own — a post,
    the views sheet, call history — records as a hold on
    the screen: opening + ending on the same frame."""
    spec = spec or TimelineSpec()
    clips = [_clip(spec.start_hold_s + spec.hold_s
                   + spec.end_hold_s, "page",
                   lambda: frame)]
    return Recording(kind=kind, spec=spec, clips=clips,
                     size=tuple(frame.size),
                     drawn=dict(drawn or {}))


# ── encode + verify ──────────────────────────────────────


def record(recording: Recording, out_path) -> dict:
    """Encode a built recording to mp4 and verify it:
    the file exists, ffprobe reads a duration, and that
    duration matches the timeline within a frame plus
    container slack.  A mismatch is a failed recording,
    never a delivered file with the wrong length."""
    try:
        from ....media.motion_studio._core import (
            MotionStudioError,
            probe_duration,
            render_sequence,
        )
    except ImportError as exc:
        raise RecordingError(
            "the video pipeline is not available on this "
            "host: " + str(exc)) from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise RecordingError(
            "numpy is not available on this host to pipe "
            "frames: " + str(exc)) from exc
    width, height = (int(recording.size[0]),
                     int(recording.size[1]))
    if width <= 0 or height <= 0 or not recording.clips:
        raise RecordingError(
            "nothing to record — the timeline is empty")
    # H.264 yuv420p needs even dimensions; pad an odd
    # edge by one replicated pixel rather than failing a
    # real screen.
    pad_w = width % 2
    pad_h = height % 2
    fps = float(recording.spec.fps)
    n_frames = max(1, int(round(
        recording.duration_s * fps)))

    def _frame(i: int, t: float):
        img = recording.frame_at(t).convert("RGB")
        if pad_w or pad_h:
            padded = Image.new(
                "RGB", (width + pad_w, height + pad_h))
            padded.paste(img, (0, 0))
            if pad_w:
                padded.paste(
                    img.crop((width - 1, 0, width, height)),
                    (width, 0))
            if pad_h:
                padded.paste(
                    img.crop((0, height - 1, width, height)),
                    (0, height))
            img = padded
        return np.asarray(img, dtype=np.uint8)

    out = Path(out_path)
    try:
        render_sequence(n_frames, width + pad_w,
                        height + pad_h, fps, out, _frame)
    except MotionStudioError as exc:
        raise RecordingError(
            "the video encode failed: " + str(exc)) from exc
    probed = probe_duration(out)
    expected = recording.duration_s
    if probed <= 0:
        raise RecordingError(
            "the encoded video probes with no duration")
    if abs(probed - expected) > 0.35:
        raise RecordingError(
            "the encoded video runs " + format(probed, ".2f")
            + "s but the timeline is "
            + format(expected, ".2f") + "s")
    return {"path": str(out), "duration_s": round(probed, 3),
            "expected_s": round(expected, 3),
            "frames": n_frames, "fps": fps,
            "size": [width + pad_w, height + pad_h],
            "kind": recording.kind}


__all__ = [
    "Clip",
    "Recording",
    "RecordingError",
    "TimelineSpec",
    "build_call_recording",
    "build_chat_recording",
    "build_comment_recording",
    "build_hold_recording",
    "build_list_recording",
    "build_profile_recording",
    "build_scroll_recording",
    "build_status_recording",
    "build_story_recording",
    "build_video_recording",
    "record",
]
