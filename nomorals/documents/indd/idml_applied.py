"""The values of an applied object style that a page item is compared
with. IDML writes some values of a page item only when they differ
from its object style.

A faithful port of ``src/idml/applied.rs`` from storytold/indd-utils
(commit 136f973, dual MIT/Apache-2.0 — see attribution.py).
"""

from __future__ import annotations


class StyleValues:
    """The effective values of an object style as the converter writes
    them: its own, then those of the styles it is based on, then the
    root ``[None]``."""

    def __init__(self):
        #: Page item attributes (``FillColor``, ``StrokeWeight``, …).
        self.item = []
        #: ``TextFramePreference`` attributes.
        self.frame = []
        #: The style's category list (chunk 0x1B92E); ``None`` without
        #: it.
        self.enabled = None
        #: Transparency attributes (chunk 0x1B92C) by attribute-list ID.
        self.transparency = []
        #: Footnote values of the text frame settings (span, minimum
        #: spacing, space between).
        self.footnote = None
        #: Baseline frame grid of the text frame settings.
        self.baseline = None
        #: Insets of the text frame settings, in stored order.
        self.insets = None

    def item_value(self, name):
        for k, v in self.item:
            if k == name:
                return v
        return None

    def transparency_value(self, attr_id):
        for i, v in self.transparency:
            if i == attr_id:
                return v
        return None

    def frame_value(self, name):
        for k, v in self.frame:
            if k == name:
                return v
        return None


def same_value(a, b):
    """Whether two IDML values are the same: equal text, or numbers
    equal to 6 significant digits (as the fidelity differ compares
    them)."""
    if a == b:
        return True
    try:
        x = float(a)
        y = float(b)
    except (TypeError, ValueError):
        return False
    return abs(x - y) <= 5e-7 * max(abs(x), abs(y))


def _add_missing(to, from_list):
    for k, v in from_list:
        if not any(n == k for n, _ in to):
            to.append((k, v))


def is_root_object_style(os):
    """Whether ``os`` is the root object style ``[None]``."""
    return os.builtin and os.name == "[None]" and os.based_on is None


def style_values(w, uid):
    """The effective values of object style ``uid``, or ``None`` if
    the document has no such style."""
    from .idml_styles import object_style_node

    if uid in w.style_values:
        return w.style_values[uid]
    styles = w.doc.object_styles
    os = styles.get(uid)
    if os is None:
        return None
    root = next((s for s in styles.values() if is_root_object_style(s)),
                None)
    out = StyleValues()
    out.enabled = os.enabled
    seen = set()
    cur = os
    while cur is not None:
        if cur.uid in seen:
            break
        seen.add(cur.uid)
        is_root = is_root_object_style(cur)
        node = object_style_node(w, cur, is_root)
        _add_missing(out.item, node.attrs)
        for attr_id, v in cur.transparency.values:
            if all(i != attr_id for i, _ in out.transparency):
                out.transparency.append((attr_id, v))
        # A style without text frame settings takes them from the
        # style it is based on.
        if cur.frame is not None:
            if out.footnote is None and cur.frame.footnotes is not None:
                span, a, b = cur.frame.footnotes
                out.footnote = (span == 1, a, b)
            if out.baseline is None:
                out.baseline = cur.frame.baseline_grid
            if out.insets is None:
                out.insets = cur.frame.insets
        if (cur.frame is not None or is_root):
            t = node.child("TextFramePreference")
            if t is not None:
                _add_missing(out.frame, t.attrs)
        nxt = cur.based_on
        if nxt is not None and nxt in styles:
            cur = styles[nxt]
        elif not is_root:
            cur = root
        else:
            cur = None
    # A chain with no text frame settings: the style's own written
    # values.
    if not out.frame:
        t = object_style_node(w, os, False).child("TextFramePreference")
        if t is not None:
            out.frame = list(t.attrs)
    w.style_values[uid] = out
    return out
