"""Bidi-safe placeholder filling — the craft-libs ``craft-i18n``
crate's ``fmt``/``isolate``/``ltr``/``strip_isolates`` (``lib.rs``),
ported.

A translated sentence with a file name in it can reorder itself
around that name when the sentence is right-to-left: a Latin file
name in an Arabic message stays where the translator put it only
when each inserted value goes in as an isolate.  This module is
that rule, exactly as upstream writes it: values are wrapped in
first-strong isolates when the text needs them, left-to-right
isolates for numbers/paths/codes, and everything a value opens
inside is closed inside its isolate — a value can never escape
and reorder the sentence around it.

The bidi classes come from :mod:`unicodedata` (``R``/``AL`` are
the strong right-to-left classes); the isolate/embedding marks
are the same code points upstream uses.
"""

from __future__ import annotations

import unicodedata

__all__ = [
    "fmt",
    "fmt_plain",
    "has_bidi_control",
    "has_rtl",
    "isolate",
    "ltr",
    "strip_isolates",
]

LRI = "\u2066"
RLI = "\u2067"
FSI = "\u2068"
PDI = "\u2069"
LRE = "\u202a"
RLE = "\u202b"
PDF = "\u202c"
LRO = "\u202d"
RLO = "\u202e"
LRM = "\u200e"
RLM = "\u200f"
ALM = "\u061c"


def isolate(value: str) -> str:
    """``value`` as a first-strong isolate (FSI…PDI): it takes the
    direction of its first strong letter and can't reorder the
    text around it.  What :func:`fmt` does to each value in text
    with a right-to-left letter."""
    return _wrap(FSI, value)


def ltr(value: str) -> str:
    """``value`` as a left-to-right isolate (LRI…PDI), for numbers,
    paths, file sizes, timecodes and codes, which read left to
    right in every language.  In an RTL first-strong isolate, the
    segments of ``/home/مستخدم/file.txt`` would run right to left.
    :func:`fmt` takes the isolate off again in text with no
    right-to-left letter."""
    return _wrap(LRI, value)


def strip_isolates(text: str) -> str:
    """``text`` without isolate marks (LRI, RLI, FSI, PDI): a
    :func:`fmt` result made fit for a file name, a log or a
    comparison."""
    return "".join(c for c in text
                   if c not in (LRI, RLI, FSI, PDI))


def _wrap(initiator: str, value: str) -> str:
    """``value`` between ``initiator`` and PDI, with every
    embedding, override and isolate it opens closed inside: a
    stray PDF or PDI is dropped, and open ones are closed in
    order.  Under the bidi algorithm the outer PDI alone would
    close them, but a renderer that handles only embeddings would
    let them run on into the rest of the text."""
    out: list[str] = [initiator]
    # One entry per open initiator: True for an isolate, False for
    # an embedding or override.
    open_stack: list[bool] = []
    for char in value:
        if char in (LRI, RLI, FSI):
            open_stack.append(True)
        elif char in (LRE, RLE, LRO, RLO):
            open_stack.append(False)
        elif char == PDF:
            if open_stack and open_stack[-1] is False:
                open_stack.pop()
            else:
                continue
        elif char == PDI:
            if True in open_stack:
                while open_stack:
                    if open_stack.pop() is False:
                        out.append(PDF)
                    else:
                        break
            else:
                continue
        out.append(char)
    while open_stack:
        out.append(PDI if open_stack.pop() else PDF)
    out.append(PDI)
    return "".join(out)


def _unwrap_isolate(value: str) -> str:
    """``value`` without its isolate, if it is exactly one."""
    if _is_one_isolate(value):
        return value[1:-1]
    return value


def _is_one_isolate(value: str) -> bool:
    """Whether ``value`` is exactly one isolate: an initiator,
    then the PDI that closes it last."""
    depth = 0
    idx = 0
    while idx < len(value):
        char = value[idx]
        idx += 1
        if char in (LRI, RLI, FSI):
            depth += 1
        elif char == PDI and depth > 0:
            depth -= 1
            if depth == 0:
                return idx == len(value)
        elif depth == 0:
            return False
    return False


def has_bidi_control(value: str) -> bool:
    """Whether ``value``, without its own isolate, has a bidi
    control: an embedding, override, isolate or mark that could
    reorder the text around it unless it is contained."""
    inner = _unwrap_isolate(value)
    if inner.isascii():
        return False
    return any(c in (LRE, RLE, PDF, LRO, RLO, LRI, RLI, FSI, PDI,
                     LRM, RLM, ALM) for c in inner)


def has_rtl(text: str) -> bool:
    """Whether ``text`` has a strong right-to-left character
    (bidi class R or AL): the text that needs isolates."""
    if text.isascii():
        return False
    return any(unicodedata.bidirectional(c) in ("R", "AL")
               for c in text)


def _fill(template: str, args: list[tuple[str, str]],
          isolated: bool) -> str:
    pieces: list[tuple[str, bool]] = []
    rest = template
    while "{" in rest:
        before, after_open = rest.split("{", 1)
        pieces.append((before, False))
        if "}" not in after_open:
            pieces.append(("{", False))
            rest = after_open
            break
        name, after_close = after_open.split("}", 1)
        match = next((v for k, v in args if k == name), None)
        if match is None:
            pieces.extend([("{", False), (name, False), ("}", False)])
        else:
            pieces.append((match, True))
        rest = after_close
    pieces.append((rest, False))
    isolated = isolated and any(
        has_rtl(text) or (is_value and has_bidi_control(text))
        for text, is_value in pieces)
    out: list[str] = []
    for text, is_value in pieces:
        if not is_value:
            out.append(text)
        elif not isolated:
            out.append(_unwrap_isolate(text))
        elif text == "":
            continue
        elif _is_one_isolate(text):
            initiator = text[0]
            out.append(_wrap(initiator, _unwrap_isolate(text)))
        else:
            out.append(isolate(text))
    return "".join(out)


def fmt(template: str, args: list[tuple[str, str]]) -> str:
    """Fill ``{name}`` placeholders from ``args`` in one pass.
    When the template or an inserted value has a right-to-left
    letter (Hebrew, Arabic and the like), or a value has a bidi
    control (an embedding, override, isolate or mark), each value
    goes in as a first-strong isolate (FSI…PDI), so its direction
    can't reorder the sentence around it.  Unknown placeholders
    stay as written, and inserted values are never read as
    templates, so a file name containing ``{n}`` stays intact."""
    return _fill(template, args, True)


def fmt_plain(template: str, args: list[tuple[str, str]]) -> str:
    """:func:`fmt` without isolates, for text that goes where the
    marks would get in the way: file names, logs, terminals, text
    compared or parsed later.  Every isolate mark is taken off, in
    the template and in the values."""
    return strip_isolates(_fill(template, args, False))
