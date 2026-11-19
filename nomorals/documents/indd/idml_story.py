"""Stories: paragraph and character ranges, anchored objects, tables,
notes, XML markers; and ``XML/BackingStory.xml`` and ``XML/Tags.xml``.

A faithful port of ``src/idml/story.rs`` from storytold/indd-utils
(commit 136f973, dual MIT/Apache-2.0 — see attribution.py).
"""

from __future__ import annotations

import bisect
import struct
from dataclasses import dataclass, field, replace

from .idml_attrs import (
    ROW_ATTRS,
    TEXT_CELL_ATTRS,
    attr_values,
    cell_attrs,
    cell_edge_attrs,
    run_attrs,
    table_attrs,
    value_text,
)
from .idml_designmap import toc_style_ref, variable_instance
from .idml_graphic import link as graphic_link
from .idml_spread import grid_data, page_item
from .idml_storylink import linked_story_option, story_link_properties, style_mappings
from .idml_styles import cell_style_ref, table_style_ref
from .idml_values import when_written as values_when_written
from .idml_xml import Xml, link_time, num, self_name, uref, write_properties, xml_tag_color
from .model import Matrix, StyleMappings


def _utf16_len(text):
    """The length of ``text`` in UTF-16 units."""
    return len(text.encode("utf-16-le")) // 2


def _utf16_units(text):
    """``text`` as a list of UTF-16 code units."""
    raw = text.encode("utf-16-le")
    return list(struct.unpack(f"<{len(raw) // 2}H", raw))


def _from_utf16(units):
    """UTF-16 code units as text, lossy as Rust's ``from_utf16_lossy``."""
    raw = struct.pack(f"<{len(units)}H", *units)
    return raw.decode("utf-16-le", errors="replace")


def _utf16_char_len(ch):
    """The length of one character in UTF-16 units."""
    return 2 if ord(ch) > 0xFFFF else 1


@dataclass
class TextState:
    """While writing text with XML markers: whether a character range
    is open, and the XML elements open inside it and around it."""

    csr: bool = False
    #: Elements open inside the character range.
    inline: list = field(default_factory=list)
    #: Elements open around character ranges.
    blocks: list = field(default_factory=list)
    #: Elements written around ranges.
    around: dict = field(default_factory=dict)
    #: A character written without its `Change` and deleted text, which
    #: were written before the text source that starts there.
    bare: int | None = None
    #: Change runs written above the character ranges (indices into the
    #: story's changes); their text has no `Change` of its own.
    lifted: list = field(default_factory=list)

    def open_csr(self, w, x, run):
        if not self.csr:
            csr_start(w, x, run)
            self.csr = True

    def close_csr(self, x):
        if self.csr:
            for _ in self.inline:
                x.end()
            self.inline.clear()
            x.end()
            self.csr = False


def _partition_point(starts, pos):
    """The first index whose start is not < ``pos`` (Rust's
    ``partition_point`` on a sorted list)."""
    return bisect.bisect_left(starts, pos)


def insertion_at_source(story, pos, length):
    """The insertion entry of a change run of inserted text that starts
    where a text source of ``length`` characters starts, at ``pos``,
    and ends before the source does."""
    starts = [c.start for c in story.changes]
    i = _partition_point(starts, pos)
    if i >= len(story.changes):
        return None
    c = story.changes[i]
    if c.start != pos or c.len >= length:
        return None
    return next((e for e in c.entries if e.is_insertion()), None)


def holding_change(story, r, offset, text_units):
    """The change run of inserted text that holds the whole extent of
    the character-level source ``r``, where the source's text in this
    run (``text_units``, from UTF-16 offset ``offset``) has no table,
    anchored item or XML marker, which a `Change` cannot hold."""
    if r.length == 0:
        return None
    end = r.start + r.length
    ends = [c.start + c.len for c in story.changes]
    i = _partition_point(ends, r.start + 1)
    if i >= len(story.changes):
        return None
    c = story.changes[i]
    if c.start > r.start or c.start + c.len < end \
            or not any(e.is_insertion() for e in c.entries):
        return None
    lo = r.start - offset
    hi = end - offset
    if lo < 0 or hi > len(text_units):
        return None
    local = text_units[lo:hi]
    blocked = any(u in (0x16, 0xFFFC) for u in local) \
        or any(r.start <= p < end for p in story.xml_markers)
    return None if blocked else i


def lifted_changes(runs, story, sources):
    """The change runs of inserted text that ``runs`` writes above the
    character ranges: their extent is not inside one run once the runs
    are split at the paragraph-level ``sources``, it holds no table,
    anchored item or XML marker, and it nests with every source (a
    source that starts with the inserted text and goes on past it keeps
    its own rule). A change over several paragraph ranges must also not
    lie inside a source."""
    if not runs:
        return []

    def run_end(run):
        return run.start + _utf16_len(run.text)

    first, last = runs[0], runs[-1]
    lo, hi = first.start, run_end(last)
    bounds = [r.start for r in runs[1:]]
    for s in sources:
        bounds.extend([s.start, s.start + s.length])
    # Offsets of tables and anchored items, in order.
    items = []
    for r in runs:
        for i, u in enumerate(_utf16_units(r.text)):
            if u in (0x16, 0xFFFC):
                items.append(r.start + i)
    out = []
    for k, c in enumerate(story.changes):
        end = c.start + c.len
        if c.len == 0 or c.start < lo or end > hi \
                or not any(e.is_insertion() for e in c.entries) \
                or not any(c.start < b < end for b in bounds):
            continue
        nests = True
        for s in sources:
            s_end = s.start + s.length
            disjoint = s_end <= c.start or end <= s.start
            holds = c.start <= s.start and s_end <= end
            inside = s.start <= c.start and end <= s_end \
                and not (s.start == c.start and c.len < s.length)
            if not (disjoint or holds or inside):
                nests = False
                break
        item_positions = [p for p in items if p >= c.start]
        blocked = any(c.start <= p < end for p in story.xml_markers) \
            or (item_positions and item_positions[0] < end)
        if nests and not blocked:
            out.append(k)
    return out


def story_start(w, x, tag, s):
    """The start tag of a `Story` or `XmlStory` with its attributes."""
    x.start(tag).attr("Self", uref(s.uid))
    # The values of every story from DOM 21.5.
    if w.dom_version() >= (21, 5):
        x.attr("AiGeneratedStory", "false") \
            .attr("AiModelName", "$ID/") \
            .attr("AiDigitalSourceType", "$ID/") \
            .attr("AiFirstParty", "false") \
            .attr("AiTextContent", "$ID/")
    # `UserText` from version 11.2, with the value every IDML has.
    v = w.doc.version
    major = v.major
    if (major, v.minor) >= (11, 2):
        x.attrs_missing(values_when_written(tag, major))
    # From DOM 13.
    if major >= 13:
        x.attr("IsEndnoteStory", str(s.is_endnote).lower())
    # The ID that links to this story name it by.
    if tag == "Story" and s.uid in w.doc.link_source_ids:
        x.attr("LinkResourceId", str(w.doc.link_source_ids[s.uid]))
    # Without chunk 0xA44C the title is `$ID/`.
    title = s.settings.title if s.settings.title is not None else "$ID/"
    x.attr("StoryTitle", title)
    # The TOC style that made the story, if any.
    toc = next((t for t in w.doc.toc_styles if t.uid == s.toc_style),
               None) if s.toc_style is not None else None
    x.attr("AppliedTOCStyle",
           toc_style_ref(toc) if toc is not None else "n")
    if s.settings.named_grid is not None:
        from .idml_designmap import named_grid_ref
        g = named_grid_ref(w, s.settings.named_grid)
        if g is not None:
            x.attr("AppliedNamedGrid", g)
    # The values every IDML has on every story.
    x.attrs_missing(w.observed(tag))


def xml_element_start(x, e):
    """The start tag of an ``XMLElement`` with its attributes."""
    x.start("XMLElement") \
        .attr("Self", e.name) \
        .attr("MarkupTag", f"XMLTag/{self_name(e.tag)}")
    if e.content is not None:
        x.attr("XMLContent", uref(e.content))
    # The attributes come first in the element.
    for name, value in e.attributes:
        x.empty("XMLAttribute", [
            ("Self", f"{e.name}XMLAttributen{name}"),
            ("Name", name),
            ("Value", value),
        ])


def _text_size_in_q(w):
    """Whether `ViewPreference` `TextSizeMeasurementUnits` is `Q`."""
    return any(v.element == "ViewPreference"
               and v.name == "TextSizeMeasurementUnits"
               and v.value == "Q" for v in w.doc.prefs.values)


def story(w, s):
    """A ``Story`` part."""
    x = Xml()
    w.package_root(x, "Story")
    story_start(w, x, "Story", s)
    story_link_properties(w, x, s)
    # Optical margin and direction from the story's chunks; without
    # them, the values every story without the chunks has, and a size
    # of 13 Q where text sizes are in Q.
    if s.settings.optical_margin is not None:
        size, align = s.settings.optical_margin
        size_text = num(size)
        alignment = {1: "true", 0: "false"}.get(align)
    elif _text_size_in_q(w):
        size_text, alignment = "9.2125984251969", "false"
    else:
        size_text, alignment = "12", "false"
    if s.settings.direction == 1:
        direction = "RightToLeftDirection"
    elif s.settings.direction is None:
        direction = "LeftToRightDirection"
    else:
        direction = None
    prefs = []
    if alignment is not None:
        prefs.append(("OpticalMarginAlignment", alignment))
    prefs.append(("OpticalMarginSize", size_text))
    # Frame type and orientation from chunk 0x2EE; without it, the
    # values of every story without the chunk. For another orientation
    # code, the orientation of the story's frames.
    frame_type, orientation = story_layout(s.settings.layout,
                                           s.orientation)
    if frame_type is not None:
        prefs.append(("FrameType", frame_type))
    if orientation == "vertical":
        prefs.append(("StoryOrientation", "Vertical"))
    elif orientation == "horizontal":
        prefs.append(("StoryOrientation", "Horizontal"))
    if direction is not None:
        prefs.append(("StoryDirection", direction))
    x.start("StoryPreference")
    for k, v in prefs:
        x.attr(k, v)
    x.attrs_missing(w.observed("Story/StoryPreference"))
    x.end()
    # The layout grid of a frame grid story.
    if frame_type == "FrameGridType" and s.settings.grid is not None:
        grid_data(w, x, s.settings.grid)
    linked_story_option(x, s)
    x.start("InCopyExportOption")
    x.attrs_missing(w.observed("Story/InCopyExportOption"))
    x.end()
    scope = uref(s.uid)
    linked = s.settings.linked
    mappings = linked.mappings if linked is not None else StyleMappings()
    style_mappings(w, x, scope, mappings)
    if linked is not None and linked.link is not None:
        graphic_link(w, x, linked.link)
    # The element whose content is the story holds all its text.
    element = None
    if s.xml_element is not None:
        e = w.doc.xml.elements.get(s.xml_element)
        if e is not None and e.content == s.uid:
            element = e
    if element is not None:
        xml_element_start(x, element)
    story_ranges(w, x, s.runs, s, scope)
    if element is not None:
        x.end()
    x.end().end()
    return x.finish()


def csr_start(w, x, r):
    """The start tag of a character range for ``r``."""
    plain, props = run_attrs(w, r.character_attrs)
    x.start("CharacterStyleRange").attr(
        "AppliedCharacterStyle",
        w.style_ref(r.character_style, False))
    for k, v in plain:
        x.attr(k, v)
    write_properties(x, props)


def text_ranges(w, x, runs, story, scope):
    """Paragraph and character ranges for ``runs``. The final paragraph
    return (a story's last, or a table cell's terminator) is not
    written. ``scope`` is the `Self` of the enclosing story or table
    cell, which prefixes the `Self` of tables inside the text."""
    ranges(w, x, runs, story, scope, False)


def story_ranges(w, x, runs, story, scope):
    """``text_ranges`` for text whose offsets are those of the story's
    changes (the story and its table cells): inserted text over several
    character ranges is written above them."""
    ranges(w, x, runs, story, scope, True)


def ranges(w, x, runs, story, scope, lift):
    def run_end(run):
        return run.start + _utf16_len(run.text)

    # Sources written around character ranges: their runs are split at
    # the source's ends.
    if runs:
        lo, hi = runs[0].start, run_end(runs[-1])
    else:
        lo, hi = 0, 0
    sources = [r for r in story.sources
               if r.paragraph and lo <= r.start < hi
               and any(t.start <= r.start < run_end(t) for t in runs)]
    lifted = lifted_changes(runs, story, sources) if lift else []
    # XML elements whose start marker and last character lie in
    # different paragraph ranges are written around paragraph ranges,
    # which split after the start marker and at the end marker. Only
    # text with the story's offsets has its markers.
    elements = xml_spans(w, runs, story) if lift else []
    paragraph_level = []
    group = []
    for i, t in enumerate(runs):
        same = i > 0 and para_of(t) == para_of(runs[i - 1])
        g = group[-1] if group else 0
        group.append(g if i == 0 or same else g + 1)

    def group_at(p):
        i = _partition_point([t.start for t in runs], p + 1) - 1
        return group[i] if i >= 0 else 0

    paragraph_level = [(k, s, e) for k, s, e in elements
                       if s + 1 < e and group_at(s) != group_at(e - 1)]
    cuts = []
    for r in sources:
        cuts.extend([r.start, r.start + r.length])
    for k in lifted:
        c = story.changes[k]
        cuts.extend([c.start, c.start + c.len])
    for _k, s, e in paragraph_level:
        cuts.extend([s + 1, e])
    if cuts:
        runs = split_runs_at(runs, cuts)
    runs = list(runs)
    last_text = None
    if runs and runs[-1].text.endswith("\r"):
        last_text = runs[-1].text[:-1]
        if last_text == "" and len(runs) > 1:
            runs.pop()
            last_text = None
    n = len(runs)

    def text_of(i):
        if last_text is not None and i + 1 == n:
            return last_text
        return runs[i].text

    # Elements within a paragraph range are written around character
    # ranges when their start marker and their last character lie in
    # different ones. Before version 11, also when they hold an element
    # with a story as content that more of their text follows: that
    # element ends its character range.
    around = {k: "paragraphs" for k, _s, _e in paragraph_level}

    def run_at(p):
        return _partition_point([t.start for t in runs], p + 1) - 1

    # Elements with a story as content: offset, and the offset of the
    # first character after it that is not an end marker.
    story_items = []
    if elements and w.doc.version.major < 11:
        for p in sorted(q for q in story.xml_markers if lo <= q < hi):
            m = story.xml_markers[p]
            if m.kind == "placeholder":
                e = w.doc.xml.elements.get(m.key)
                if e is not None and e.story_content:
                    nxt = p + 1
                    while story.xml_markers.get(nxt) is not None \
                            and story.xml_markers[nxt].kind == "end":
                        nxt += 1
                    story_items.append((p, nxt))
    for k, s, e in elements:
        starts = [q for q, _n in story_items]
        from_i = _partition_point(starts, s + 1)
        ends_range = any(nxt < e for _q, nxt in story_items[from_i:]
                         if _q < e)
        if k not in around and (run_at(s) != run_at(e - 1) or ends_range):
            around[k] = "characters"
    # Paragraph-level elements open at a paragraph range that starts
    # after their start marker, and close before one that starts at
    # their end marker.
    element_starts = {s + 1: (k, e) for k, s, e in paragraph_level}
    element_ends = {e for _k, _s, e in paragraph_level}
    open_elements = []
    # A lifted change over runs of more than one paragraph range is a
    # child of the story, the others of their paragraph range.
    group2 = [0] * n
    for i in range(1, n):
        group2[i] = group2[i - 1] \
            + (1 if para_of(runs[i]) != para_of(runs[i - 1]) else 0)

    def change_end(k):
        c = story.changes[k]
        return c.start + c.len

    top, inner = [], []
    for k in lifted:
        c = story.changes[k]
        inside = [group2[i] for i in range(n)
                  if runs[i].start >= c.start
                  and run_end(runs[i]) <= change_end(k)]
        (top if inside and inside[0] != inside[-1] else inner).append(k)

    def insertion(k):
        return next((e for e in story.changes[k].entries
                     if e.is_insertion()), None)

    def boundary(p):
        return any(story.changes[k].start == p or change_end(k) == p
                   for k in top) \
            or p in element_starts or p in element_ends

    # Story-level changes open, innermost last.
    outer = []
    i = 0
    while i < n:
        p = runs[i].start
        if open_elements and open_elements[-1][1] == p:
            open_elements.pop()
            x.end()
        if p in element_starts:
            k, e = element_starts[p]
            el = w.doc.xml.elements.get(k)
            if el is not None:
                xml_element_start(x, el)
                open_elements.append((k, e))
        # A paragraph-level element's end marker alone between two
        # others is in no paragraph range.
        if runs[i].text == "\uFEFF" \
                and story.xml_markers.get(p) is not None \
                and story.xml_markers[p].kind == "end" \
                and around.get(story.xml_markers[p].key) == "paragraphs" \
                and (i + 1 == n or boundary(runs[i + 1].start)):
            i += 1
            continue
        starting = [k for k in top if story.changes[k].start == runs[i].start]
        starting.sort(key=lambda k: -change_end(k))
        for k in starting:
            x.start("Change")
            e = insertion(k)
            if e is not None:
                change_attrs(w, x, e, "InsertedText")
            outer.append(k)
        para = para_of(runs[i])
        plain, props = run_attrs(w, para[1])
        x.start("ParagraphStyleRange") \
            .attr("AppliedParagraphStyle", w.style_ref(para[0], True))
        for k, v in plain:
            x.attr(k, v)
        write_properties(x, props)
        st = TextState(lifted=list(lifted), around=dict(around))
        # Paragraph-level sources and changes open, innermost last,
        # with their end offsets.
        open_ends = []
        first = i
        while i < n and para_of(runs[i]) == para \
                and (i == first or not boundary(runs[i].start)):
            r = runs[i]
            # Sources and changes that start here, the longest first;
            # a change before a source of the same extent.
            starting = [(change_end(k), True, k) for k in inner
                        if story.changes[k].start == r.start]
            starting.extend(
                (s.start + s.length, False, j)
                for j, s in enumerate(sources) if s.start == r.start)
            starting.sort(key=lambda t: (-t[0], not t[1]))
            for end, is_change, k in starting:
                if is_change:
                    x.start("Change")
                    e = insertion(k)
                    if e is not None:
                        change_attrs(w, x, e, "InsertedText")
                    open_ends.append(end)
                    continue
                s = sources[k]
                src = w.doc.text_sources.get(s.source)
                if src is None:
                    continue
                # A source that starts with inserted text and goes on
                # past it follows a character range with the deleted
                # text there and an empty `Change`.
                e = insertion_at_source(story, r.start, s.length)
                if e is not None:
                    csr_start(w, x, r)
                    deleted_text(w, x, story, r.start, scope)
                    x.start("Change")
                    change_attrs(w, x, e, "InsertedText")
                    x.end()
                    x.end()
                    st.bare = r.start
                source_start(w, x, src)
                open_ends.append(end)
            # An end marker is in no character range.
            end_first = r.text.startswith("\uFEFF") \
                and story.xml_markers.get(r.start) is not None \
                and story.xml_markers[r.start].kind == "end"
            if not end_first:
                csr_start(w, x, r)
                st.csr = True
            run_content(w, x, text_of(i), r, story, scope, st)
            st.close_csr(x)
            while open_ends and open_ends[-1] <= run_end(r):
                x.end()
                open_ends.pop()
            i += 1
        for _ in open_ends:
            x.end()
        # Elements left open (their end is missing) end with the range.
        for _ in st.blocks:
            x.end()
        st.blocks.clear()
        x.end()
        end = run_end(runs[i - 1])
        while outer and change_end(outer[-1]) <= end:
            x.end()
            outer.pop()
    for _ in outer:
        x.end()
    for _ in open_elements:
        x.end()


def xml_spans(w, runs, story):
    """The XML elements written in ``runs`` whose start and end markers
    both lie in them: key, start and end offsets, in marker order."""
    if not runs:
        return []
    first, last = runs[0], runs[-1]
    hi = last.start + _utf16_len(last.text)

    def inside(p):
        i = _partition_point([t.start for t in runs], p + 1)
        return i > 0 and p < runs[i - 1].start \
            + _utf16_len(runs[i - 1].text)

    starts = {}
    out = []
    for p in sorted(q for q in story.xml_markers
                    if first.start <= q < hi):
        m = story.xml_markers[p]
        if m.kind == "start" and inside(p) \
                and m.key in w.doc.xml.elements:
            starts[m.key] = p
        elif m.kind == "end" and inside(p) and m.key in starts:
            out.append((m.key, starts.pop(m.key), p))
    out.sort(key=lambda t: t[1])
    return out


def run_content(w, x, text, run, story, scope, st):
    """Content, line breaks, anchored items, tables and XML elements of
    one character range ``run``, whose text is ``text``. A character
    range is open on entry (``st.csr``); XML markers can close it and
    open others."""
    offset = run.start
    buf = []

    def flush():
        if buf:
            # INDD stores a forced line break as LF; IDML as U+2028.
            content = "".join(buf).replace("\n", "\u2028")
            x.start("Content").text(content).end()
            buf.clear()

    pos = offset
    # End offset of the open hyperlink text source.
    open_end = None
    # End offset of the open endnote range.
    endnote = None
    # The open `Change` of inserted text: the index of its change run.
    # It is always the innermost open element, so the others can close
    # around it; it is opened again where its text continues.
    inserted = None

    def close_inserted():
        nonlocal inserted
        if inserted is not None:
            inserted = None
            flush()
            x.end()

    # The `Change` of inserted text that holds the open text source:
    # the index of its change run. It closes after the source.
    holding = None
    utf16 = _utf16_units(text)
    for ch in text:
        if open_end is not None and open_end == pos:
            close_inserted()
            flush()
            x.end()
            open_end = None
            # The holding change goes on as the innermost element
            # where its text continues.
            if holding is not None:
                i = holding
                holding = None
                c = story.changes[i]
                if c.start + c.len > pos:
                    inserted = i
                else:
                    x.end()
        if endnote is not None and endnote == pos:
            close_inserted()
            flush()
            if open_end is not None:
                open_end = None
                x.end()
            if holding is not None:
                holding = None
                x.end()
            x.end()
            endnote = None
        if endnote is None:
            r = next((r for r in story.endnote_ranges if r.start == pos),
                     None)
            if r is not None:
                # Around the endnote's text, inside the character range.
                close_inserted()
                flush()
                if open_end is not None:
                    open_end = None
                    x.end()
                if holding is not None:
                    holding = None
                    x.end()
                st.open_csr(w, x, run)
                x.start("EndnoteRange") \
                    .attr("Self", uref(r.uid)) \
                    .attr("SourceEndnote", uref(r.endnote))
                endnote = r.start + r.len
        # Deleted text of tracked changes, before the character that
        # follows it.
        if st.bare != pos and pos in story.deletions:
            close_inserted()
            flush()
            st.open_csr(w, x, run)
            deleted_text(w, x, story, pos, scope)
        # A character-level text source opens before its first
        # character, which can be an element (a note, a text
        # destination). The schema allows no paragraph destination in
        # it: one at the start stays before the source.
        paragraph_destination = any(
            d.kind[0] == "paragraph"
            for d in story.text_destinations.get(pos, []))
        if open_end is None and endnote is None \
                and pos not in story.xml_markers \
                and not paragraph_destination:
            r = next((r for r in story.sources
                      if r.start == pos and not r.paragraph), None)
            if r is not None:
                src = w.doc.text_sources.get(r.source)
                if src is not None:
                    at_source = insertion_at_source(story, pos, r.length)
                    # Inserted text that holds the whole source: the
                    # source is inside its `Change`.
                    hold = None
                    if at_source is None:
                        hold = holding_change(story, r, offset, utf16)
                        if hold is not None and hold in st.lifted:
                            hold = None
                    flush()
                    if hold is not None and inserted == hold:
                        inserted = None
                        holding = hold
                    else:
                        close_inserted()
                    st.open_csr(w, x, run)
                    # A source that starts with inserted text and goes
                    # on past it follows an empty `Change`, and its
                    # first character has none.
                    if at_source is not None:
                        x.start("Change")
                        change_attrs(w, x, at_source, "InsertedText")
                        x.end()
                        st.bare = pos
                    elif hold is not None and holding is None:
                        x.start("Change")
                        e = next((e for e in story.changes[hold].entries
                                  if e.is_insertion()), None)
                        if e is not None:
                            change_attrs(w, x, e, "InsertedText")
                        holding = hold
                    source_start(w, x, src)
                    open_end = r.start + r.length
        # Inserted text: inside a `Change`, except a table, an anchored
        # item and an XML marker, which close it (the schema allows no
        # page item in a `Change`).
        starts = [c.start + c.len for c in story.changes]
        want = _partition_point(starts, pos + 1)
        want_i = None
        if want < len(story.changes) \
                and story.changes[want].start <= pos \
                and any(e.is_insertion()
                        for e in story.changes[want].entries):
            c = story.changes[want]
            if ch not in ("\x16", "\uFFFC") \
                    and pos not in story.xml_markers \
                    and st.bare != pos \
                    and holding != want \
                    and want not in st.lifted:
                want_i = want
        if inserted != want_i:
            close_inserted()
            if want_i is not None:
                e = next((e for e in story.changes[want_i].entries
                          if e.is_insertion()), None)
                if e is not None:
                    flush()
                    st.open_csr(w, x, run)
                    x.start("Change")
                    change_attrs(w, x, e, "InsertedText")
                    inserted = want_i
        # An index marker is written as a `PageReference` in place of
        # its character.
        if ch == "\uFEFF" and pos not in story.xml_markers \
                and pos in story.index_markers:
            m = story.index_markers[pos]
            flush()
            st.open_csr(w, x, run)
            x.start("PageReference").attr("Self", uref(m.uid))
            if m.current_page:
                x.attr("PageReferenceType", "CurrentPage")
            t = w.topics.get(m.uid)
            if t is not None:
                x.attr("ReferencedTopic", t)
            if m.id is not None:
                x.attr("Id", str(m.id))
            x.end()
            pos += 1
            continue
        if ch == "\uFEFF" and pos in story.notes:
            flush()
            st.open_csr(w, x, run)
            note(w, x, story.notes[pos], story, scope)
            pos += 1
            continue
        if ch == "\uFEFF" and pos in story.text_destinations:
            # Written as empty elements in place of the character.
            flush()
            st.open_csr(w, x, run)
            for d in story.text_destinations[pos]:
                x.start("ParagraphDestination"
                        if d.kind[0] == "paragraph"
                        else "HyperlinkTextDestination") \
                    .attr("Self", d.reference()) \
                    .attr("Name", d.name) \
                    .attr("Hidden", str(d.hidden).lower())
                if d.key is not None:
                    x.attr("DestinationUniqueKey", str(d.key))
                x.end()
            if pos not in story.xml_markers:
                if paragraph_destination and open_end is None \
                        and endnote is None:
                    r = next((r for r in story.sources
                              if r.start == pos and not r.paragraph),
                             None)
                    if r is not None:
                        src = w.doc.text_sources.get(r.source)
                        if src is not None:
                            close_inserted()
                            source_start(w, x, src)
                            open_end = r.start + r.length
                pos += 1
                continue
        if ch == "\uFEFF" and pos in story.xml_markers:
            flush()
            close_inserted()
            # An XML marker ends the open text source.
            if open_end is not None:
                open_end = None
                x.end()
            if holding is not None:
                holding = None
                x.end()
            m = story.xml_markers[pos]
            if m.kind == "start":
                # An element written around ranges has its start marker
                # in a character range of its parent, and its end
                # marker in none.
                e = w.doc.xml.elements.get(m.key)
                around = st.around.get(m.key)
                if e is not None and around == "paragraphs":
                    st.open_csr(w, x, run)
                elif e is not None and around == "characters":
                    st.open_csr(w, x, run)
                    st.close_csr(x)
                    xml_element_start(x, e)
                    st.blocks.append(m.key)
                elif e is not None:
                    st.open_csr(w, x, run)
                    xml_element_start(x, e)
                    st.inline.append(m.key)
            elif m.kind == "end":
                if st.inline and st.inline[-1] == m.key:
                    st.inline.pop()
                    x.end()
                elif st.blocks and st.blocks[-1] == m.key:
                    st.close_csr(x)
                    st.blocks.pop()
                    x.end()
            elif m.kind == "placeholder":
                e = w.doc.xml.elements.get(m.key)
                if e is not None:
                    st.open_csr(w, x, run)
                    xml_element_start(x, e)
                    x.end()
                    # Before version 11, an element with a story as
                    # content ends its character range.
                    if e.story_content and w.doc.version.major < 11:
                        st.close_csr(x)
            # Hidden markers write nothing.
            pos += 1
            continue
        st.open_csr(w, x, run)
        if ch == "\r":
            flush()
            x.start("Br").end()
        elif ch == "\uFFFC" and pos in story.anchors:
            flush()
            for item in story.anchors[pos]:
                page_item(w, x, item, False, Matrix.IDENTITY)
        elif ch == "\x04" and pos in story.endnotes:
            e_uid, r_uid = story.endnotes[pos]
            flush()
            x.start("Endnote") \
                .attr("Self", uref(e_uid)) \
                .attr("EndnoteTextRange", uref(r_uid)) \
                .end()
        elif ch == "\x04" and pos in story.footnotes:
            # The footnote's text, in place of its reference.
            flush()
            x.start("Footnote")
            text_ranges(w, x, story.footnotes[pos].runs, story, scope)
            x.end()
        elif ch == "\x16" and pos in story.tables:
            flush()
            table(w, x, story.tables[pos], story, scope)
        elif ch == "\x17":
            # Internal table markers that follow U+0016.
            pass
        elif ch == "\x18" and pos in story.text_variables:
            flush()
            variable_instance(w, x, story.text_variables[pos], story.uid)
        else:
            buf.append(ch)
        pos += _utf16_char_len(ch)
    flush()
    close_inserted()
    if open_end is not None:
        x.end()
    if holding is not None:
        x.end()
    if endnote is not None:
        x.end()


def deleted_text(w, x, story, pos, scope):
    """The deleted text of tracked changes owned by the character at
    ``pos``, each in a `Change`."""
    ds = story.deletions.get(pos)
    if ds is None:
        return
    entries = [e for c in story.changes if c.start == pos
               for e in c.entries if e.is_deletion()]
    it = iter(entries)
    for d in ds:
        x.start("Change")
        e = next(it, None)
        if e is not None:
            change_attrs(w, x, e, "DeletedText")
        else:
            x.attr("ChangeType", "DeletedText")
        text_ranges(w, x, d.runs, story, scope)
        x.end()


def change_attrs(w, x, e, kind):
    """The attributes of a `Change`, in IDML's order."""
    name, user = document_user(w, e.user)
    x.attr("Date", link_time(e.time, w.doc.xmp_dates)) \
        .attr("ChangeType", kind) \
        .attr("UserName", name) \
        .attr("AppliedDocumentUser", user)


def note(w, x, n, story, scope):
    """A `Note` with its text."""
    name, user = document_user(w, n.user)
    x.start("Note") \
        .attr("Collapsed", "false") \
        .attr("CreationDate", link_time(n.created, w.doc.xmp_dates)) \
        .attr("ModificationDate", link_time(n.modified, w.doc.xmp_dates)) \
        .attr("UserName", name) \
        .attr("AppliedDocumentUser", user)
    text_ranges(w, x, n.runs, story, scope)
    x.end()


def document_user(w, name):
    """`UserName` and `AppliedDocumentUser` for a user name stored with
    a note or a change: the first document user with that name (`n` if
    none), whose name IDML replaces when it is the unknown user."""
    for i, (flag, u) in enumerate(w.doc.users):
        if u == name:
            shown = "$ID/Unknown User Name" if flag == 2 else name
            return shown, f"dDocumentUser{i:x}"
    return name, "n"


def source_start(w, x, src):
    """The start tag of a text source, `HyperlinkTextSource` or
    `CrossReferenceSource`, with its attributes and properties."""
    style = w.style_ref(src.character_style, False) \
        if src.character_style is not None else "n"
    if src.format is not None:
        x.start("CrossReferenceSource") \
            .attr("Self", uref(src.uid)) \
            .attr("AppliedFormat", uref(src.format))
    else:
        x.start("HyperlinkTextSource") \
            .attr("Self", uref(src.uid))
    x.attr("Name", src.name) \
        .attr("Hidden", str(src.hidden).lower()) \
        .attr("AppliedCharacterStyle", style)
    alternative_destination(x, src.alternative)


def alternative_destination(x, a):
    """`Properties/AlternativeDestination` of a text source, for the
    one type seen."""
    if a is None or not a.is_toc_anchor():
        return
    # A control character in the page number is written as a
    # processing instruction in the text of the attribute.
    page = ""
    for c in a.page_number:
        if ord(c) < 0x20:
            page += f"<?AID {ord(c):04x}?>"
        else:
            page += c
    x.start("Properties") \
        .start("AlternativeDestination") \
        .attr("Type", "TocTextAnchor") \
        .attr("IndexMarkerId", str(a.index_marker)) \
        .attr("TextAnchorName", a.anchor_name) \
        .attr("TocEntryPageNumberString", page) \
        .attr("TocEntryLevel", str(a.level)) \
        .end() \
        .end()


def text_cell_values(w, value):
    """The text cell values IDML writes on every table, row, column
    and cell from DOM 11, with the value in effect."""
    if w.doc.version.major < 11:
        return []
    out = []
    for attr_id, name, kind in TEXT_CELL_ATTRS:
        v = value(attr_id)
        if v is not None:
            text = value_text(w, kind, v)
            if text is not None:
                out.append((name, text))
    return out


def table(w, x, t, story, scope):
    """A table in story text."""
    from .model import TableStyles

    styles = TableStyles(w.doc.cell_styles, w.doc.table_styles)
    ident = f"{scope}i{t.uid:x}"
    rows = len(t.rows)
    x.start("Table") \
        .attr("Self", ident) \
        .attr("HeaderRowCount", str(t.header_rows)) \
        .attr("FooterRowCount", str(t.footer_rows)) \
        .attr("BodyRowCount",
              str(max(rows - (t.header_rows + t.footer_rows), 0))) \
        .attr("ColumnCount", str(len(t.columns)))
    st = t.style
    if st is not None:
        style = w.doc.table_styles.get(st)
        if style is not None:
            x.attr("AppliedTableStyle",
                   table_style_ref(w, "TableStyle", style))
    x.attr("TableDirection",
           "RightToLeftDirection" if t.right_to_left
           else "LeftToRightDirection")
    for name, v in table_attrs(w, t.attrs, False):
        x.attr(name, v)
    for name, v in text_cell_values(w, lambda a: t.value(a, styles)):
        x.attr(name, v)
    # A row (column) has the values of the first cell that starts in
    # it.

    def first_cell(of):
        c = next((c for c in t.cells if of(c)), None)
        if c is not None:
            return text_cell_values(
                w, lambda a: t.cell_value(c, a, styles))
        return text_cell_values(w, lambda a: t.value(a, styles))

    for i, r in enumerate(t.rows):
        x.start("Row") \
            .attr("Self", f"{ident}Row{i:x}") \
            .attr("Name", str(i))
        for name, v in first_cell(lambda c, i=i: c.row == i):
            x.attr(name, v)
        if r.height is not None:
            x.attr("SingleRowHeight", num(r.height))
        if r.min_height is not None:
            x.attr("MinimumHeight", num(r.min_height))
        for name, v in attr_values(w, r.attrs, ROW_ATTRS):
            x.attr(name, v)
        x.end()
    for i, width in enumerate(t.columns):
        x.start("Column") \
            .attr("Self", f"{ident}Column{i:x}") \
            .attr("Name", str(i))
        for name, v in first_cell(lambda c, i=i: c.column == i):
            x.attr(name, v)
        x.attr("SingleColumnWidth", num(width))
        x.end()
    for c in t.cells:
        cell_id = f"{ident}i{c.id:x}"
        x.start("Cell") \
            .attr("Self", cell_id) \
            .attr("Name", f"{c.column}:{c.row}") \
            .attr("RowSpan", str(c.row_span)) \
            .attr("ColumnSpan", str(c.column_span))
        # From DOM 11.
        if w.doc.version.major >= 11:
            x.attr("CellType",
                   "TextTypeCell" if c.kind == "text"
                   else "GraphicTypeCell")
        for name, v in text_cell_values(
                w, lambda a, c=c: t.cell_value(c, a, styles)):
            x.attr(name, v)
        for name, v in cell_edge_attrs(w, t, c):
            x.attr(name, v)
        f = t.format(c)
        if f is not None:
            for name, v in cell_attrs(w, f.attrs):
                x.attr(name, v)
            r = cell_style_ref(w, f.style)
            if r is not None:
                x.attr("AppliedCellStyle", r) \
                    .attr("AppliedCellStylePriority",
                          str(f.style_priority))
        else:
            # A cell without an attribute set has no cell style.
            x.attr("AppliedCellStyle", "CellStyle/$ID/[None]") \
                .attr("AppliedCellStylePriority", "0")
        story_ranges(w, x, c.runs, story, cell_id)
        # IDML writes a graphic cell's item like a nested item,
        # without `Locked`.
        for item in c.items:
            page_item(w, x, item, True, Matrix.IDENTITY)
        # Deleted text owned by the cell's terminator, which no range
        # holds, ends the cell.
        if c.runs and c.runs[-1].text.endswith("\r"):
            end = c.runs[-1].start + _utf16_len(c.runs[-1].text) - 1
            deleted_text(w, x, story, end, cell_id)
        x.end()
    x.end()


def backing_story(w):
    """``XML/BackingStory.xml``."""
    x = Xml()
    w.package_root(x, "BackingStory")
    s = w.doc.xml.story
    if s is not None:
        story_start(w, x, "XmlStory", s)
        story_ranges(w, x, s.runs, s, uref(s.uid))
        x.end()
    x.end()
    return x.finish()


def tags(w):
    """``XML/Tags.xml``."""
    x = Xml()
    w.package_root(x, "Tags")
    # IDML lists the tags by name, ignoring case.
    ordered = sorted(w.doc.xml_tags, key=lambda t: t[0].lower())
    for name, color in ordered:
        x.start("XMLTag") \
            .attr("Self", f"XMLTag/{self_name(name)}") \
            .attr("Name", name)
        if color is not None:
            c = xml_tag_color(color)
            if c is not None:
                write_properties(x, [("TagColor", "enumeration",
                                      ("text", c))])
        x.end()
    x.end()
    return x.finish()


def para_of(t):
    """The paragraph formatting of a run and its paragraph-style
    record: runs with the same ones share a paragraph range."""
    return (t.paragraph_style, t.paragraph_attrs, t.paragraph_run)


def split_runs_at(runs, cuts):
    """``runs`` with each run that holds one of ``cuts`` strictly
    inside split there into two runs with the same styles."""
    out = []
    for r in runs:
        units = _utf16_units(r.text)
        end = r.start + len(units)
        at = sorted({c for c in cuts if r.start < c < end})
        from_off = r.start
        for c in at + [end]:
            out.append(replace(
                r, start=from_off,
                text=_from_utf16(units[from_off - r.start:c - r.start])))
            from_off = c
    return out


def story_layout(layout, frames):
    """`FrameType` and the orientation of a story from chunk 0x2EE (u16
    orientation, u16 frame type); without the chunk, `TextFrameType`
    and horizontal. For another orientation code, the orientation of
    the story's frames."""
    orientation, frame_type = layout if layout is not None else (0, 0)
    if frame_type == 0:
        frame_name = "TextFrameType"
    elif frame_type == 1:
        frame_name = "FrameGridType"
    else:
        frame_name = None
    if orientation == 0:
        orientation_name = "horizontal"
    elif orientation == 1:
        orientation_name = "vertical"
    else:
        orientation_name = frames
    return frame_name, orientation_name
