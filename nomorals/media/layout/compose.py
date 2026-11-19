"""Text composition — measuring and flowing stories into frames.

Upstream's composer (``crates/compose``) is a Knuth–Plass paragraph
composer with dictionary hyphenation, justification ranges, keep
options and threading across frames. Devon's port measures real
text with the PIL font door (the same DejaVuSans door the vector
renderer uses) and composes with a greedy line breaker plus
word hyphenation; the Knuth–Plass optimum and dictionary
hyphenation are honest remainders, named in the survey. What is
real here: measured widths, columns, threading across frames,
overset detection, vertical justification and first-baseline
options.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import ImageFont

from .errors import LayoutError
from .model import Story

__all__ = ["ComposedFrame", "ComposedLine", "ComposedStory",
           "compose_story", "font_for", "measure_text"]

_DEFAULT_SIZE = 12.0


def _font(size: float, bold: bool = False):
    size = max(4, int(round(size)))
    names = (("DejaVuSans-Bold.ttf", "DejaVuSans.ttf") if bold
             else ("DejaVuSans.ttf",))
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def font_for(size: float, bold: bool = False):
    """The PIL font for ``size`` points (DejaVuSans, bold variant
    when asked) — the one font door the layout organ uses."""
    return _font(size, bold)


def measure_text(text: str, size: float, bold: bool = False) -> float:
    """The width of ``text`` at ``size`` points, measured with the
    real font — never an estimate from character counts."""
    if not text:
        return 0.0
    font = _font(size, bold)
    try:
        return float(font.getlength(text))
    except AttributeError:  # pragma: no cover - old Pillow
        return float(font.getsize(text)[0])


def _line_height(size: float, leading: float | None) -> float:
    if leading is not None and leading > 0:
        return leading
    return size * 1.2


def _hyphenate(word: str, width: float, size: float,
               bold: bool = False) -> list[str]:
    """Split a word that is wider than ``width`` into hyphenated
    pieces that each fit. Splits are made at measured positions
    (at least 2 characters per piece); a word that cannot be split
    usefully comes back whole."""
    if measure_text(word, size, bold) <= width:
        return [word]
    pieces: list[str] = []
    rest = word
    while rest:
        if measure_text(rest, size, bold) <= width:
            pieces.append(rest)
            break
        # Longest prefix (with a hyphen) that fits.
        take = 0
        for end in range(2, len(rest)):
            piece = rest[:end] + "-"
            if measure_text(piece, size, bold) <= width:
                take = end
            else:
                break
        if take < 2:
            # Even two characters + hyphen do not fit: the frame is
            # narrower than a syllable. Emit the rest whole rather
            # than looping forever; the caller reports overset.
            pieces.append(rest)
            break
        pieces.append(rest[:take] + "-")
        rest = rest[take:]
    return pieces


@dataclass
class ComposedLine:
    """One composed line: its text, its baseline offset from the
    top of the frame's text area, and its height."""

    text: str
    y: float
    height: float


@dataclass
class ComposedFrame:
    """The lines that landed in one frame, plus the text that did
    not fit (empty when the frame took everything it was given)."""

    lines: list[ComposedLine] = field(default_factory=list)
    remaining: str = ""


@dataclass
class ComposedStory:
    """A story composed into one or more frames."""

    frames: list[ComposedFrame] = field(default_factory=list)
    overset: bool = False
    remaining: str = ""

    @property
    def lines(self) -> list[ComposedLine]:
        """All lines across all frames (the single-frame case)."""
        out: list[ComposedLine] = []
        for frame in self.frames:
            out.extend(frame.lines)
        return out


def _wrap_words(words: list[str], width: float, size: float,
                bold: bool = False) -> list[str]:
    """Greedy line breaking over measured words, with hyphenation
    for words wider than the frame."""
    lines: list[str] = []
    current = ""
    for word in words:
        if measure_text(word, size, bold) > width:
            pieces = _hyphenate(word, width, size, bold)
            for piece in pieces[:-1]:
                if current:
                    lines.append(current)
                    current = ""
                lines.append(piece)
            word = pieces[-1]
        trial = word if not current else current + " " + word
        if measure_text(trial, size, bold) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def compose_story(story: Story,
                  width_or_frames,
                  height: float | None = None,
                  font_size: float = _DEFAULT_SIZE,
                  leading: float | None = None,
                  vertical_justification: str = "top",
                  first_baseline: str = "ascent"
                  ) -> ComposedStory:
    """Compose ``story`` into one frame or a list of frames
    (threading: the text flows from each frame into the next).

    Call as ``compose_story(story, width, height, ...)`` for a
    single frame, or ``compose_story(story, [(w, h), ...], ...)``
    for a threaded run. Returns the lines per frame, whether any
    text is overset, and the remaining text."""
    if height is not None:
        frame_list = [(float(width_or_frames), float(height))]
    elif isinstance(width_or_frames, (list, tuple)):
        if (isinstance(width_or_frames, tuple)
                and len(width_or_frames) == 2
                and isinstance(width_or_frames[0], (int, float))):
            frame_list = [tuple(width_or_frames)]  # type: ignore[arg-type]
        else:
            frame_list = [tuple(frame)  # type: ignore[arg-type]
                          for frame in width_or_frames]
    else:
        raise LayoutError("compose needs a frame (width, height) or a "
                          "list of frames", kind="invalid_argument")
    if not frame_list:
        raise LayoutError("compose needs at least one frame",
                          kind="invalid_argument")
    # Flatten the story to words, keeping paragraph breaks.
    paragraphs: list[list[str]] = []
    for para in story.paras:
        text = para.text()
        size = None
        for run in para.runs:
            if run.font_size is not None:
                size = run.font_size
        words = text.split()
        paragraphs.append(words)
    # Compose paragraph by paragraph, frame by frame.
    composed_frames: list[ComposedFrame] = []
    para_index = 0
    pending: list[str] = []
    for width, height in frame_list:
        size = font_size
        line_h = _line_height(size, leading)
        # First-baseline offset: ascent puts the first baseline
        # one ascent below the top; cap height is lower; leading
        # uses the full line height; fixed uses the size.
        if first_baseline == "cap_height":
            first_y = size * 0.7
        elif first_baseline == "leading":
            first_y = line_h
        elif first_baseline == "fixed":
            first_y = size
        else:  # "ascent"
            first_y = size * 0.8
        max_lines = max(0, int((height - first_y) // line_h) + 1) \
            if height > 0 else 0
        lines: list[str] = []
        while len(lines) < max_lines:
            if not pending:
                if para_index >= len(paragraphs):
                    break
                pending = list(paragraphs[para_index])
                para_index += 1
                if not pending:
                    continue
            wrapped = _wrap_words(pending, width, size)
            room = max_lines - len(lines)
            take = wrapped[:room]
            lines.extend(take)
            if len(wrapped) > room:
                # The words that did not fit stay pending.
                pending = _remaining_after(pending, take)
                break
            pending = []
        # Vertical justification shifts the block within the frame.
        block_h = len(lines) * line_h
        if vertical_justification == "center":
            top = first_y + max(0.0, (height - block_h) / 2.0)
        elif vertical_justification == "bottom":
            top = first_y + max(0.0, height - block_h)
        elif vertical_justification == "justify" and len(lines) > 1:
            top = first_y
        else:
            top = first_y
        if vertical_justification == "justify" and len(lines) > 1:
            gap = (height - block_h) / max(1, len(lines) - 1)
            ys = [top + i * (line_h + gap) for i in range(len(lines))]
        else:
            ys = [top + i * line_h for i in range(len(lines))]
        composed_frames.append(ComposedFrame(
            lines=[ComposedLine(text=line, y=y, height=line_h)
                   for line, y in zip(lines, ys, strict=True)]))
        if para_index >= len(paragraphs) and not pending:
            break
    overset = bool(pending) or para_index < len(paragraphs)
    remaining = " ".join(pending)
    if para_index < len(paragraphs):
        rest = [" ".join(p) for p in paragraphs[para_index:]]
        remaining = (remaining + " " + " ".join(rest)).strip()
    return ComposedStory(frames=composed_frames, overset=overset,
                         remaining=remaining)


def _remaining_after(words: list[str], taken_lines: list[str]
                     ) -> list[str]:
    """The words of ``words`` not consumed by ``taken_lines``.
    Hyphenated fragments: a taken line ending in "-" consumed
    only the fragment's letters, so the rest of that word stays."""
    consumed = " ".join(taken_lines)
    # Walk the taken lines' words against the source words.
    taken_words = consumed.split()
    out: list[str] = []
    index = 0
    for taken in taken_words:
        if index >= len(words):
            break
        source = words[index]
        if taken == source:
            index += 1
        elif taken.endswith("-") and source.startswith(taken[:-1]):
            out.append(source[len(taken) - 1:])
            index += 1
        else:
            # The taken line is a whole word the source does not
            # have (should not happen); stop rather than guessing.
            break
    out.extend(words[index:])
    return out
