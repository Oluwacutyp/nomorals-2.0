"""Spreads and master spreads: pages, guides, page items with their
paths, frame settings, text wrap and fitting, and placed graphics.

A faithful port of ``src/idml/spread.rs`` from storytold/indd-utils
(commit 136f973, dual MIT/Apache-2.0 — see attribution.py). Helpers
that upstream defines in other writer modules live there:
``carries_ink``/``reference_carries_ink`` in idml_styles,
``graphic_contour``/``placed_graphic``/``link`` in idml_graphic,
``media`` in idml_media, ``timing_setting``/``animation_setting`` in
idml_animation, ``object_export_option`` in idml_export.
"""

from __future__ import annotations

from .idml import number_style
from .idml_applied import is_root_object_style, same_value, style_values
from .idml_attrs import (
    ANCHORED_SETTINGS,
    CONTOUR_PARAMS,
    anchored_settings,
    contour_type,
    dash_written,
    fitting_attrs,
    gradients,
    item_attr_values,
    layout_constraints,
    text_wrap_mode,
    text_wrap_offsets,
    text_wrap_side,
)
from .idml_values import Node, element, when_written
from .idml_xml import (
    Xml,
    base64_lines,
    frame_grid_color,
    matrix,
    num,
    nums,
    page_color,
    set_property,
    uref,
    write_properties,
)
from .ids import Numbering
from .model import BaselineGrid
from .object import builtin_key


def spread_origin(pages):
    """The top-left corner of a spread's pages in spread coordinates:
    the smallest left and top edge of any page (0, 0 without pages)."""
    corners = []
    for p in pages:
        a, b, c, d, tx, ty = p.transform.values
        left, top = p.bounds[0], p.bounds[1]
        corners.append((a * left + c * top + tx,
                        b * left + d * top + ty))
    if not corners:
        return (0.0, 0.0)
    return (min(c[0] for c in corners), min(c[1] for c in corners))


def _pages_extent(pages):
    """The bounding box of a spread's pages (min x, min y, max x, max
    y), from the corners of each page through its transform."""
    out = None
    for p in pages:
        a, b, c, d, tx, ty = p.transform.values
        left, top, right, bottom = p.bounds
        for x, y in [(left, top), (right, top),
                     (left, bottom), (right, bottom)]:
            px, py = a * x + c * y + tx, b * x + d * y + ty
            if out is None:
                out = [px, py, px, py]
            else:
                out[0] = min(out[0], px)
                out[1] = min(out[1], py)
                out[2] = max(out[2], px)
                out[3] = max(out[3], py)
    return out


def _form_tag(kind):
    return {
        "button": "Button",
        "check_box": "CheckBox",
        "radio_button": "RadioButton",
        "text_box": "TextBox",
        "combo_box": "ComboBox",
        "signature_field": "SignatureField",
        "multi_state": "MultiStateObject",
    }[kind]


def _state_type(state_id):
    return {0: "Up", 1: "Rollover", 2: "Down", 3: "UpOn", 4: "UpOff",
            6: "RolloverOff", 7: "DownOn", 8: "DownOff"}.get(state_id)


#: Page item attributes that IDML writes only when they differ from
#: the applied object style.
STYLE_COMPARED = [
    "FillColor", "FillTint", "StrokeColor", "StrokeWeight", "MiterLimit",
    "CornerOption", "CornerRadius", "TopLeftCornerRadius",
    "TopRightCornerRadius", "BottomLeftCornerRadius",
    "BottomRightCornerRadius", "TopLeftCornerOption",
    "TopRightCornerOption", "BottomLeftCornerOption",
    "BottomRightCornerOption", "StrokeType", "StrokeAlignment",
    "StrokeTint", "StrokeDashAndGap", "StrokeCornerAdjustment",
    "GapColor", "GapTint", "EndCap", "EndJoin", "LeftLineEnd",
    "RightLineEnd", "ArrowHeadAlignment",
]

#: An overprint attribute with the IDs of its colour, the IDML name of
#: the colour, the IDs of the tint and its name, and the ID of the flag.
OVERPRINT = [
    ("OverprintFill", 0x6E68, "FillColor", 0x6E69, "FillTint", 0x6E6A),
    ("OverprintStroke", 0x6E64, "StrokeColor", 0x6E66, "StrokeTint", 0x6E67),
]


def _category_off(name, on):
    """Whether an object style whose category list is ``on`` turns off
    the category of a fill, stroke or corner attribute."""
    def off(i):
        return i not in on

    corners = off(0x1B935) and off(0x1B936)
    if name in ("FillColor", "FillTint", "OverprintFill"):
        return off(0x1B933)
    if name in ("StrokeColor", "OverprintStroke"):
        return off(0x1B934)
    if name in ("StrokeType", "StrokeTint", "GapColor", "GapTint"):
        return off(0x1B934) and corners
    return corners


def _full_tint(item, name):
    """Whether the item stores a tint (``FillTint``, ``StrokeTint``,
    ``GapTint``) of 100. IDML writes it as −1, but it differs from a
    style value of −1."""
    attr_id = {"FillTint": 0x6E69, "StrokeTint": 0x6E66,
               "GapTint": 0x6E8A}.get(name)
    if attr_id is None:
        return False
    v = item.attrs.get(attr_id)
    return v is not None and v.as_f64() == 100.0


def _frame_category(name):
    """The category of a ``TextFramePreference`` attribute: the ID that
    turns it on in an object style's category list."""
    if name in ("TextColumnCount", "TextColumnGutter",
                "TextColumnFixedWidth", "UseFixedColumnWidth",
                "UseFlexibleColumnWidth", "TextColumnMaxWidth",
                "VerticalJustification", "VerticalThreshold",
                "IgnoreWrap", "VerticalBalanceColumns"):
        return 0x1B93E
    if name in ("FirstBaselineOffset", "MinimumFirstBaselineOffset"):
        return 0xADC8
    if name.startswith("Footnotes"):
        return 0xADCA
    if name.startswith("ColumnRule"):
        return 0xADCB
    if "AutoSizing" in name:
        return 0xADC9
    return None


def _frame_always(name, major):
    """``TextFramePreference`` attributes that IDML writes on every
    frame of the version, whatever the object style."""
    if name == "TextColumnMaxWidth":
        return major >= 8
    if name == "TextColumnCount":
        return major >= 10
    return False


def _plain_rectangle(paths):
    """Whether ``paths`` is a plain rectangle: one path of exactly four
    straight points joined by horizontal and vertical sides in the
    frame's inner coordinates."""
    if len(paths) != 1:
        return False
    pts = paths[0].points
    if len(pts) != 4:
        return False
    if not all(p.left == p.anchor and p.right == p.anchor for p in pts):
        return False
    def eq(a, b):
        return abs(a - b) <= 1e-4

    for i in range(4):
        a = pts[i].anchor
        b = pts[(i + 1) % 4].anchor
        if not (eq(a[0], b[0]) or eq(a[1], b[1])):
            return False
    return True


#: ``AutoSizingReferencePoint`` codes of frames and object styles.
POINTS = ["TopLeftPoint", "TopCenterPoint", "TopRightPoint",
          "LeftCenterPoint", "CenterPoint", "RightCenterPoint",
          "BottomLeftPoint", "BottomCenterPoint", "BottomRightPoint"]
#: ``PageTransitionType`` codes of spreads.
TRANSITION_TYPES = [(0, "None"), (0x205C8, "BlindsTransition"),
                    (0x205CA, "CombTransition"), (0x205CC, "DissolveTransition"),
                    (0x205CF, "PushTransition"), (0x205D0, "SplitTransition")]
#: ``PageTransitionDirection`` codes of spreads.
TRANSITION_DIRECTIONS = [(0, "NotApplicable"), (1, "Down"),
                         (0xB, "Horizontal"), (0xD, "HorizontalIn")]
#: ``BaselineFrameGridRelativeOption`` codes.
BASELINE_RELATIVE = ["TopOfPage", "TopOfMargin", "TopOfFrame", "TopOfInset"]
#: ``VerticalJustification`` codes.
JUSTIFY = ["TopAlign", "CenterAlign", "BottomAlign", "JustifyAlign"]
#: ``FirstBaselineOffset`` codes.
FIRST_BASELINE = ["LeadingOffset", "AscentOffset", "CapHeight",
                  "EmboxHeight", "XHeight"]
#: ``AutoSizingType`` codes.
SIZING = ["Off", "HeightOnly", "WidthOnly", "HeightAndWidth",
          "HeightAndWidthProportionally"]


def text_path(w, x, t):
    """A ``TextPath``."""
    x.start("TextPath") \
        .attr("Self", uref(t.uid)) \
        .attr("ParentStory", uref(t.story))
    if t.default_codes:
        x.attr("PathAlignment", "CenterPathAlignment") \
            .attr("TextAlignment", "BaselineTextAlignment") \
            .attr("PathEffect", "RainbowPathEffect")
    x.attr("FlipPathEffect", "Flipped" if t.flipped else "NotFlipped")
    if t.default_codes:
        x.attr("PathSpacing", "0")
    x.attr("StartBracket", num(t.start)) \
        .attr("EndBracket", num(t.end)) \
        .attr("PreviousTextFrame", uref(t.previous)) \
        .attr("NextTextFrame", uref(t.next))
    x.end()


def _eps_text_properties(x, e):
    """The ``Properties`` of EPS text: ``PathBoundingBox``,
    ``EPSTextData`` and ``EPSTextAttributeBounds``."""
    left, top, right, bottom = e.path_bounds
    x.start("Properties")
    x.empty("PathBoundingBox", [
        ("Left", num(left)),
        ("Top", num(top)),
        ("Right", num(right)),
        ("Bottom", num(bottom)),
    ])
    x.start("EPSTextData") \
        .cdata(base64_lines(e.data), 262144) \
        .end()
    left, top, right, bottom = e.attr_bounds
    x.empty("EPSTextAttributeBounds", [
        ("Top", num(top)),
        ("Left", num(left)),
        ("Bottom", num(bottom)),
        ("Right", num(right)),
    ])
    x.end()


def label(x, pairs):
    """A ``Label`` element with one ``KeyValuePair`` per pair; nothing
    without pairs."""
    if not pairs:
        return
    x.start("Label")
    for k, v in pairs:
        x.empty("KeyValuePair", [("Key", k), ("Value", v)])
    x.end()


def path_geometry_element(x, paths):
    """The ``PathGeometry`` element of ``paths``, without
    ``Properties``."""
    geometry_paths(x, "PathGeometry", paths)


def geometry_paths(x, tag, paths):
    """Element ``tag`` with one ``GeometryPathType`` per path."""
    x.start(tag)
    for p in paths:
        x.start("GeometryPathType") \
            .attr("PathOpen", str(p.open).lower())
        x.start("PathPointArray")
        for pt in p.points:
            x.empty("PathPointType", [
                ("Anchor", nums([pt.anchor[0], pt.anchor[1]])),
                ("LeftDirection", nums([pt.left[0], pt.left[1]])),
                ("RightDirection", nums([pt.right[0], pt.right[1]])),
            ])
        x.end().end()
    x.end()


def text_frame_preference(w, x, p, style, plain):
    """``TextFramePreference`` of a text frame. IDML writes an
    attribute only when its category is off in the frame's object
    style or its value differs from the style's, apart from a few
    attributes it always writes. ``style`` is the applied style's
    values; without a style every value is written. ``plain`` is
    whether the frame is a plain rectangle without a corner option."""
    major = w.doc.version.major

    def swatch(u):
        return "n" if u == 0 else w.doc.swatches.get(u)

    attrs = [
        ("TextColumnCount", str(p.column_count)),
        ("TextColumnGutter", num(p.column_gutter)),
        ("TextColumnFixedWidth", num(p.column_fixed_width)),
    ]
    if 0 <= p.first_baseline_offset < len(FIRST_BASELINE):
        attrs.append(("FirstBaselineOffset",
                      FIRST_BASELINE[p.first_baseline_offset]))
    if 0 <= p.vertical_justification < len(JUSTIFY):
        attrs.append(("VerticalJustification",
                      JUSTIFY[p.vertical_justification]))
    attrs.append(("VerticalBalanceColumns",
                  str(p.vertical_balance_columns).lower()))
    if p.auto_sizing_type is not None \
            and 0 <= p.auto_sizing_type < len(SIZING):
        attrs.append(("AutoSizingType", SIZING[p.auto_sizing_type]))
    if p.auto_sizing_reference_point is not None \
            and 0 <= p.auto_sizing_reference_point < len(POINTS):
        attrs.append(("AutoSizingReferencePoint",
                      POINTS[p.auto_sizing_reference_point]))
    if p.max_width is not None:
        attrs.append(("TextColumnMaxWidth", num(p.max_width)))
    attrs.append(("UseFixedColumnWidth", str(p.use_fixed_width).lower()))
    if p.minimum_sizes is not None:
        (use_height, use_width), (height, width), no_breaks = p.minimum_sizes
        attrs.append(("UseMinimumHeightForAutoSizing", str(use_height).lower()))
        attrs.append(("MinimumHeightForAutoSizing", num(height)))
        attrs.append(("UseMinimumWidthForAutoSizing", str(use_width).lower()))
        attrs.append(("MinimumWidthForAutoSizing", num(width)))
        attrs.append(("UseNoLineBreaksForAutoSizing", str(no_breaks).lower()))
    # Without the chunk, `false`.
    attrs.append(("IgnoreWrap",
                  str(p.ignore_wrap if p.ignore_wrap is not None else False).lower()))
    # 0 on every frame of the samples; IDML writes it by the baseline
    # category rule below.
    attrs.append(("MinimumFirstBaselineOffset", "0"))
    # Values every IDML of the version has where it writes them; they
    # also say from which version the column rule and footnote settings
    # exist. Footnote values from 13.1.
    written = when_written("TextFrame/TextFramePreference", major)

    def has(k):
        return any(n == k for n, _ in written)

    footnotes = w.saved_by((13, 1))
    if has("ColumnRuleOffset"):
        if p.column_rule is not None:
            width, color = p.column_rule[0], swatch(p.column_rule[1])
        else:
            # Without the chunk.
            width, color = 1.0, "Color/Black"
        attrs.append(("ColumnRuleStrokeWidth", num(width)))
        if color is not None:
            attrs.append(("ColumnRuleStrokeColor", color))
        attrs.append(("ColumnRuleOverride",
                      str(p.column_rule_override is True).lower()))
    if has("FootnotesEnableOverrides") and footnotes:
        spacing, between = p.footnotes if p.footnotes is not None else [12.0, 6.0]
        attrs.append(("FootnotesSpanAcrossColumns", str(p.footnote_span).lower()))
        attrs.append(("FootnotesMinimumSpacing", num(spacing)))
        attrs.append(("FootnotesSpaceBetween", num(between)))
    for k, val in written:
        if not footnotes and k.startswith("Footnotes"):
            continue
        if not any(n == k for n, _ in attrs):
            attrs.append((k, val))
    if style is not None:
        def keep(kv):
            k, val = kv
            if _frame_always(k, major):
                return True
            cat = _frame_category(k)
            if cat is not None and style.enabled is not None \
                    and cat in style.enabled:
                sv = style.frame_value(k)
                if sv is not None and same_value(sv, val):
                    return False
            return True

        attrs = [kv for kv in attrs if keep(kv)]
    # A category is off when the applied style's list lacks its ID, or
    # the frame has no style.
    def off(i):
        return style is None or style.enabled is None \
            or i not in style.enabled

    footnote_options = _frame_footnote_options(w, p, style, off(0xADCA))
    baseline = _frame_baseline_grid(w, p, style, off(0xADC8))
    # Before DOM 11 a frame that is not a plain rectangle always has
    # insets; those of a plain rectangle follow the rule of the general
    # attributes: written where they differ from the style's, in any
    # order, or the style turns the category off.
    def sorted4(v):
        return sorted(v)

    insets = major >= 11 or not plain or off(0x1B93E) \
        or (style is not None and sorted4(style.insets or [0.0] * 4)
            != sorted4(p.inset))
    # Before DOM 11 IDML gives a frame without values no element.
    if attrs or insets:
        x.start("TextFramePreference")
        for k, val in attrs:
            x.attr(k, val)
        if insets:
            top, left, bottom, right = p.inset

            def item(tag, v):
                return Node(tag=tag, attrs=[("type", "unit")],
                            text=num(v))

            # A frame that is not a plain rectangle has one number,
            # whatever its four insets.
            if not plain:
                node = item("InsetSpacing", p.inset_single)
            else:
                node = Node(tag="InsetSpacing", attrs=[("type", "list")],
                            children=[item("ListItem", v)
                                      for v in (top, left, bottom, right)])
            write_properties(x, [], [node])
        x.end()
    if baseline is not None:
        baseline.write(x)
    if footnote_options is not None:
        x.empty("TextFrameFootnoteOptionsObject", footnote_options)


def _frame_footnote_options(w, p, style, off):
    """``TextFrameFootnoteOptionsObject`` of a text frame: from DOM 12,
    when the footnote category is off or the frame's footnote values
    differ from its style's."""
    if w.doc.version.major < 12:
        return None
    # Without chunk 0x22608 the values are false, 12 and 6.
    min_v, between = p.footnotes if p.footnotes is not None else [12.0, 6.0]
    span, smin, sbetween = style.footnote if style is not None \
        and style.footnote is not None else (False, 12.0, 6.0)
    differ = p.footnote_span != span \
        or not same_value(num(min_v), num(smin)) \
        or not same_value(num(between), num(sbetween))
    if off or differ:
        return [
            ("EnableOverrides", "false"),
            ("SpanFootnotesAcross", str(p.footnote_span).lower()),
            ("MinimumSpacingOption", num(min_v)),
            ("SpaceBetweenFootnotes", num(between)),
        ]
    return None


def _frame_baseline_grid(w, p, style, all_values):
    """``BaselineFrameGridOption`` of a text frame: every value when
    the baseline category is off, else the values that differ from the
    style's; ``None`` when there is nothing to write."""
    g = p.baseline_grid if p.baseline_grid is not None else BaselineGrid.DEFAULT
    sg = style.baseline if style is not None and style.baseline is not None \
        else BaselineGrid.DEFAULT
    n = Node(tag="BaselineFrameGridOption")

    def push(differs, k, v):
        if all_values or differs:
            n.attrs.append((k, v))

    push(g.use_custom != sg.use_custom, "UseCustomBaselineFrameGrid",
         str(g.use_custom).lower())
    push(not same_value(num(g.start), num(sg.start)),
         "StartingOffsetForBaselineFrameGrid", num(g.start))
    if 0 <= g.relative < len(BASELINE_RELATIVE):
        push(g.relative != sg.relative, "BaselineFrameGridRelativeOption",
             BASELINE_RELATIVE[g.relative])
    push(not same_value(num(g.increment), num(sg.increment)),
         "BaselineFrameGridIncrement", num(g.increment))
    if all_values or g.color != sg.color:
        if g.color == 0:
            color = None
            rgb = w.doc.prefs.baseline_frame_grid_color
            if rgb is not None:
                color = frame_grid_color(rgb)
        else:
            color = None
            if p.baseline_rgb is not None:
                from .idml_xml import ui_color_property
                color = ui_color_property("BaselineFrameGridColor",
                                          p.baseline_rgb)
        if color is not None:
            set_property(n, color)
    if n.attrs or n.children:
        return n
    return None


def text_wrap_preference(x, wrap, contour):
    """``TextWrapPreference`` from an item's text wrap chunk and
    contour. An item without the chunk has no wrap. Values not
    identified yet are left out: the whole element for an unknown
    mode, and the side and inverse settings for flags other than 1."""
    mode = text_wrap_mode(wrap)
    if mode is None:
        return
    x.start("TextWrapPreference")
    # `Inverse` is false on every item of the samples.
    x.attr("Inverse", "false")
    if wrap is None:
        x.attr("ApplyToMasterPageOnly", "false") \
            .attr("TextWrapSide", "BothSides")
    else:
        x.attr("ApplyToMasterPageOnly", str(wrap.master_flag == 0).lower())
        side = text_wrap_side(wrap.side)
        if side is not None:
            x.attr("TextWrapSide", side)
    x.attr("TextWrapMode", mode)
    x.start("Properties")
    # IDML has the wrap path only when it is custom; the schema puts
    # it before `TextWrapOffset`.
    if wrap is not None and wrap.mode != 0 and wrap.custom_path:
        path_geometry_element(x, wrap.custom_path)
    x.empty("TextWrapOffset", text_wrap_offsets(wrap))
    x.end()
    if contour is not None:
        # A type code without a sample leaves the contour out.
        typed = None
        unknown = False
        if contour.contour is not None:
            t = contour_type(contour.contour.kind)
            if t is None:
                unknown = True
            else:
                typed = (t, contour.contour.inside_edges)
        if not unknown:
            x.start("ContourOption")
            if typed is not None:
                t, inside = typed
                x.attr("ContourType", t)
                if inside is not None:
                    x.attr("IncludeInsideEdges", str(inside).lower())
            x.attr("ContourPathName", contour.path_name)
            if contour.params and contour.contour is not None:
                for k, v in CONTOUR_PARAMS:
                    x.attr(k, v)
            x.end()
    x.end()


def item_settings(w, x, item, nested):
    """The name, visibility, lock and the other settings every page
    item has; ``nested`` for an item inside another page item, which
    IDML writes without ``Locked``."""
    settings(w, x, item.props, None if nested else item.props.locked)
    for name, v in gradients(item):
        if not x.has_attr(name):
            x.attr(name, v)


def settings(w, x, p, locked):
    """The settings of ``ItemProps`` (page items and placed graphics):
    ``Locked`` only when ``locked`` is given."""
    if p.name is not None:
        name = builtin_key(p.name.name) if p.name.builtin else p.name.name
    else:
        name = "$ID/"
    x.attr("Name", name) \
        .attr("Visible", str(not p.hidden).lower())
    if locked is not None:
        x.attr("Locked", str(locked).lower())
    if p.allow_overrides is not None:
        x.attr("AllowOverrides", str(p.allow_overrides).lower())
    if w.doc.version.major >= 8:
        for name, counts in zip(
                ["ParentInterfaceChangeCount",
                 "TargetInterfaceChangeCount",
                 "LastUpdatedInterfaceChangeCount"],
                p.change_counts, strict=True):
            x.attr(name, " ".join(str(c) for c in counts))
        if p.overridden is not None and p.overridden[0] != 0:
            overridden = " ".join(str(c) for c in p.overridden[1])
        else:
            overridden = ""
        x.attr("OverriddenPageItemProps", overridden)
        lc = layout_constraints(p.layout_constraints
                                if p.layout_constraints is not None else 0x22)
        if lc is not None:
            h, v = lc
            x.attr("HorizontalLayoutConstraints", h) \
                .attr("VerticalLayoutConstraints", v)


def object_style_ref(w, uid):
    """The ``AppliedObjectStyle`` reference of an object style UID."""
    os = w.doc.object_styles.get(uid)
    if os is None:
        return None
    name = w.grouped_name(os.uid, os.name, os.builtin)
    from .idml_xml import self_name
    return f"ObjectStyle/{self_name(name)}"


def page_item(w, x, item, nested, outer):
    """A page item. ``nested`` for an item inside another page item;
    ``outer`` is the transform from the item's parent to the spread
    (for anchored items, to the anchor)."""
    from .idml_animation import animation_setting, timing_setting
    from .idml_export import object_export_option
    from .idml_graphic import graphic_contour, placed_graphic
    from .idml_media import media as write_media
    from .idml_transparency import write as transparency_write

    kind = item.kind
    if kind[0] == "text_frame":
        story = kind[1].get("story")
        if story is not None and any(t.uid == story and t.is_endnote
                                     for t in w.doc.stories):
            tag = "EndnoteTextFrame"
        else:
            tag = "TextFrame"
    elif kind[0] == "group":
        tag = "Group"
    elif kind[0] == "shape":
        shape = kind[1]
        tag = {"rectangle": "Rectangle", "oval": "Oval",
               "polygon": "Polygon", "graphic_line": "GraphicLine"}.get(
                   shape, "Rectangle")
    elif kind[0] == "eps_text":
        tag = "EPSText"
    elif kind[0] == "form":
        tag = _form_tag(kind[1].kind)
    else:
        tag = "Rectangle"
    x.start(tag).attr("Self", uref(item.uid))
    if item.article_children is not None:
        kept = [u for u in item.article_children
                if any(c.uid == u for c in item.children)]
        dropped = [u for u in item.article_children
                   if not any(c.uid == u for c in item.children)]
        if dropped:
            w.warn(f"group {item.uid}: article children {dropped} left "
                   f"out: not written in the group")
        x.attr("ArticleChildrenList",
               " ".join(uref(u) for u in kept))
    if kind[0] == "text_frame":
        d = kind[1]
        x.attr("ParentStory", uref(d.get("story"))) \
            .attr("PreviousTextFrame", uref(d.get("previous"))) \
            .attr("NextTextFrame", uref(d.get("next"))) \
            .attr("ContentType", "TextType")
        # The ID that links to this frame name it by.
        link_id = w.doc.link_source_ids.get(item.uid)
        if link_id is not None:
            x.attr("LinkResourceId", str(link_id))
    elif kind[0] == "shape":
        content = "Unassigned" if not item.graphics \
            and not item.props.graphic_frame else "GraphicType"
        x.attr("ContentType", content)
    applied = style_values(w, item.object_style) \
        if item.object_style is not None else None
    # A category that the item's object style (not the root) turns
    # off: IDML writes the item's values of that category even where
    # they equal the style's.
    on = None
    if applied is not None and item.object_style is not None:
        os = w.doc.object_styles.get(item.object_style)
        if os is not None and not is_root_object_style(os):
            on = applied.enabled
    def forced(name):
        return on is not None and _category_off(name, on)

    # A form field's fill, stroke and corner values are not those of
    # its attribute list.
    form = kind[0] == "form"
    own = [] if form else item_attr_values(w, item.attrs)
    # The stroke type in effect decides the dash attributes.
    stroke_type = None
    if tag != "Group":
        for n, v, _full in values_in_effect(w, item):
            if n == "StrokeType":
                stroke_type = v
                break
    for name, v in own:
        if not dash_written(name, stroke_type):
            continue
        # Fill, stroke and corner values equal to the object style's
        # are left out.
        same = name in STYLE_COMPARED and not forced(name) \
            and not _full_tint(item, name) \
            and applied is not None \
            and applied.item_value(name) is not None \
            and same_value(applied.item_value(name), v)
        if not same:
            x.attr(name, v)
    # A fill, stroke or corner attribute that the item does not store
    # has the value of the document's base list; IDML writes it where
    # it differs from the object style's, and always for an item
    # without an object style. Groups vary.
    if tag != "Group" and not form \
            and (applied is not None or item.object_style is None) \
            and w.doc.prefs.item_base is not None:
        base = w.doc.prefs.item_base
        rest_values = [(i, v) for i, v in base.values
                       if item.attrs.get(i) is None]
        from .attrs import Attrs
        rest = Attrs()
        rest.values = rest_values
        for name, v in item_attr_values(w, rest):
            # IDML writes no corner attributes for EPS text.
            if (tag == "EPSText" and "Corner" in name) \
                    or not dash_written(name, stroke_type):
                continue
            if applied is not None:
                s = applied.item_value(name)
                write = s is not None and (forced(name)
                                           or not same_value(s, v))
            else:
                write = True
            if name in STYLE_COMPARED and write:
                x.attr(name, v)
    # A group stores none of these values; IDML gives it the value
    # all its children have in effect.
    if tag == "Group":
        shared = shared_values(w, item)
        for name in STYLE_COMPARED:
            found = next(((v, f) for n, v, f in shared if n == name), None)
            if found is None:
                continue
            v, full = found
            s = applied.item_value(name) if applied is not None else None
            write = forced(name) or full or s is None \
                or not same_value(s, v)
            if write:
                x.attr(name, v)
    if tag == "Group":
        for name, _c, _cn, _t, _tn, _f in OVERPRINT:
            style = applied.item_value(name) if applied is not None else None
            got = overprint_in_effect(w, item,
                                      next(o for o in OVERPRINT if o[0] == name))
            if got is not None and got[0] is not None \
                    and str(got[0]).lower() != (style or "false"):
                x.attr(name, str(got[0]).lower())
    elif not form:
        _item_overprint(w, x, item, applied, forced)
    # An item without an object style refers to none, `n`. Where a
    # form field with states keeps its style is not known.
    ref = object_style_ref(w, item.object_style) \
        if item.object_style is not None else None
    if not form or ref is not None:
        x.attr("AppliedObjectStyle", ref if ref is not None else "n")
    # Only items directly on a spread have `ItemLayer`.
    if item.layer is not None and not nested:
        x.attr("ItemLayer", uref(item.layer))
    x.attr("ItemTransform", matrix(item.transform))
    # From 21.4; -1 without the chunk.
    if w.saved_by((21, 4)):
        if item.props.grouping_layer is not None:
            x.attr("BeforeGroupingLayerReference",
                   uref(item.props.grouping_layer))
        x.attr("BeforeGroupingLayerPosition",
               str(item.props.grouping_layer_position
                   if item.props.grouping_layer_position is not None
                   else -1))
    item_settings(w, x, item, nested)
    if form:
        _form_attrs(x, kind[1])
    # An endnote text frame has the values of a text frame.
    observed = "TextFrame" if tag == "EndnoteTextFrame" else tag
    x.attrs_missing(w.observed(observed))
    # A form field has the bounds of what it shows in place of paths.
    if form:
        bounds, paths = kind[1].bounds, []
    else:
        bounds, paths = None, item.paths
    if item.label or bounds is not None or paths:
        x.start("Properties")
        label(x, item.label)
        if bounds is not None:
            left, top, right, bottom = bounds
            x.empty("PathBoundingBox", [
                ("Left", num(left)),
                ("Top", num(top)),
                ("Right", num(right)),
                ("Bottom", num(bottom)),
            ])
        if paths:
            path_geometry_element(x, paths)
        x.end()
    # A frame grid.
    if kind[0] == "text_frame" and kind[1].get("preferences") is not None \
            and kind[1]["preferences"].frame_grid:
        count = None
        for tag_name, _g, c in w.doc.prefs.grids:
            if tag_name == "StoryGridDataInformation":
                count = c
                break
        if count is not None:
            x.empty("GridDataInformation", [
                ("GridView", "GridViewEnum"),
                ("CharacterCountLocation", "BottomAlign"),
                ("CharacterCountSize", num(count)),
            ])
    for t in item.text_paths:
        text_path(w, x, t)
    if kind[0] == "eps_text":
        _eps_text_properties(x, kind[1])
    prefs = kind[1].get("preferences") if kind[0] == "text_frame" else None
    if prefs is not None:
        plain = _plain_rectangle(item.paths) and all(
            not n.endswith("CornerOption") or v == "None"
            for n, v, _ in values_in_effect(w, item))
        text_frame_preference(w, x, prefs, applied, plain)
    frame = kind[0] == "shape" and kind[1] in ("rectangle", "oval", "polygon")
    if frame:
        frame_fitting(w, x, item)
    # EPS text and form fields have no export options.
    if kind[0] not in ("eps_text", "form"):
        n = object_export_option(
            w.doc.version, item.export, item.props.export_size_type,
            item.props.generated_alt,
            "group" if tag == "Group" else "item")
        if n is not None:
            n.write(x)
    # IDML writes the anchor settings that differ from the object
    # style's, which is not known for a form field with states; all of
    # them where the style turns the category off.
    unknown_style = form and item.object_style is None
    # An item inside an anchored item has the anchor's settings.
    inherited = None
    if nested and item.anchor is None and w.anchors:
        inherited = w.anchors[-1]
    anchor = item.anchor if item.anchor is not None else inherited
    if anchor is not None and not unknown_style:
        style = w.doc.object_styles.get(item.object_style) \
            if item.object_style is not None else None
        off = style is not None and style.enabled is not None \
            and 0xCA2F not in style.enabled
        style_anchor = style.anchor if style is not None else None
        if off:
            style_anchor = None
        attrs = [a for a in anchored_settings(anchor)
                 if style_anchor is None
                 or a not in anchored_settings(style_anchor)]
        # A complete element has two more values.
        if all(any(k == n for k, _ in attrs) for n in ANCHORED_SETTINGS):
            attrs.append(("LockPosition", "false"))
            attrs.append(("AnchorSpaceAbove", "0"))
        if attrs:
            x.empty("AnchoredObjectSetting", attrs)
    # A frame has the contour of its graphic, unless its object style
    # turns the text wrap category on.
    style_wrap = on is not None and all(i in on for i in (0x1B942, 0x37C8, 0x37C9))
    contour = None
    if len(item.graphics) == 1:
        contour = graphic_contour(w, item.graphics[0])
        if style_wrap and contour is not None:
            from .idml_graphic import ContourOption
            contour = ContourOption(contour=None, **{
                "path_name": contour.path_name,
                "params": contour.params})
    text_wrap_preference(x, item.text_wrap, contour)
    if frame:
        # The values every IDML has.
        observed = w.observed(f"{tag}/InCopyExportOption")
        if observed:
            x.start("InCopyExportOption")
            x.attrs_missing(observed)
            x.end()
    for effect, attr, v in transparency_write(
            w, x, item.attrs, applied, uref(item.uid)):
        w.warn(f"item {item.uid}: {effect} {attr} {v} is outside the "
               f"IDML range; left out")
    # IDML writes a multi-state object's animation after its states;
    # the schema allows a button's only before its timing and states.
    multi_state = kind[0] == "form" and kind[1].kind == "multi_state"
    if not multi_state and item.animation is not None:
        animation_setting(w, x, item.uid, item.animation)
    if form:
        timing_setting(w, x, uref(item.uid), kind[1].timing)
    if item.anchor is not None:
        w.anchors.append(item.anchor)
    for child in item.children:
        page_item(w, x, child, True, item.transform.then(outer))
    if item.anchor is not None:
        w.anchors.pop()
    if form:
        _form_states(w, x, item, kind[1], item.transform.then(outer))
        _form_behaviors(w, x, item, kind[1])
    if multi_state and item.animation is not None:
        animation_setting(w, x, item.uid, item.animation)
    for g in item.graphics:
        placed_graphic(w, x, g, item.transform.then(outer))
    for m in item.media:
        write_media(w, x, m, item.transform.then(outer))
    x.end()


def values_in_effect(w, item):
    """The fill, stroke and corner values (``STYLE_COMPARED``) a page
    item has in effect, with the stored-100 tint flag: its own value,
    else the document's base list value where IDML writes it, else its
    object style's. A group has the value all its children share; a
    form field has none."""
    kind = item.kind
    if kind[0] == "group":
        return shared_values(w, item)
    if kind[0] == "form":
        return []
    applied = style_values(w, item.object_style) \
        if item.object_style is not None else None
    on = None
    if applied is not None and item.object_style is not None:
        os = w.doc.object_styles.get(item.object_style)
        if os is not None and not is_root_object_style(os):
            on = applied.enabled
    own = item_attr_values(w, item.attrs)
    base = []
    if w.doc.prefs.item_base is not None and \
            (applied is not None or item.object_style is None):
        b = w.doc.prefs.item_base
        from .attrs import Attrs
        rest = Attrs()
        rest.values = [(i, v) for i, v in b.values
                       if item.attrs.get(i) is None]
        base = item_attr_values(w, rest)
    out = []
    for name in STYLE_COMPARED:
        if kind[0] == "eps_text" and "Corner" in name:
            continue
        # StrokeType comes before the dash attributes in the list.
        stroke = next((v for n, v, _ in out if n == "StrokeType"), None)
        if not dash_written(name, stroke):
            continue
        style = applied.item_value(name) if applied is not None else None
        found = next(((n, v) for n, v in own if n == name), None)
        if found is not None:
            out.append((name, found[1], _full_tint(item, name)))
            continue
        found = next(((n, v) for n, v in base if n == name), None)
        if found is not None:
            forced = on is not None and _category_off(name, on)
            written = applied is None or (style is not None and
                                          (forced or not same_value(style, found[1])))
            if written:
                out.append((name, found[1], False))
                continue
        if style is not None:
            out.append((name, style, False))
    return out


def overprint_in_effect(w, item, overprint):
    """The overprint flag of ``overprint`` (an ``OVERPRINT`` entry)
    that a page item has in effect: its own flag, else its object
    style's value, else ``False``. Before 21.4 an item whose fill or
    stroke puts no ink on the page has none (``(None,)``). A group has
    the flag that all its children with one share, ``(None,)`` when
    none has one; ``None`` when they differ, and for a form field."""

    name, color_id, color_name, tint_id, tint_name, flag_id = overprint
    if item.kind[0] == "group":
        shared = None
        for c in item.children:
            got = overprint_in_effect(w, c, overprint)
            if got is None:
                return None
            v = got[0]
            if v is None:
                continue
            if shared is None:
                shared = v
            elif v != shared:
                return None
        return (shared,)
    if item.kind[0] == "form":
        return None
    applied = style_values(w, item.object_style) \
        if item.object_style is not None else None
    if not w.saved_by((21, 4)) and not _puts_ink(w, item, applied, overprint):
        return (None,)
    v = item.attrs.get(flag_id)
    if v is not None:
        u = v.as_u32()
        if u is not None and 0 <= u <= 1:
            return (u == 1,)
        return None
    style_val = applied.item_value(name) if applied is not None else None
    return (style_val == "true",)


def shared_values(w, group):
    """The values of ``values_in_effect`` that all children of
    ``group`` have. EPS text has no corner values and is left out for
    those."""
    children = [(c.kind[0] == "eps_text", values_in_effect(w, c))
                for c in group.children]
    out = []
    for name in STYLE_COMPARED:
        shared = None
        all_have = True
        any_have = False
        for eps, values in children:
            if eps and "Corner" in name:
                continue
            any_have = True
            found = next(((v, f) for n, v, f in values if n == name), None)
            if found is None:
                all_have = False
            elif shared is None:
                shared = found
            elif same_value(shared[0], found[0]):
                shared = (shared[0], shared[1] and found[1])
            else:
                all_have = False
        if all_have and any_have and shared is not None:
            out.append((name, shared[0], shared[1]))
    return out


def _puts_ink(w, item, applied, overprint):
    """Whether the fill or stroke of ``overprint`` puts ink on the
    page: the colour (the item's, else its object style's) carries ink
    and the tint is not 0."""
    from .idml_styles import carries_ink, reference_carries_ink

    _name, color_id, color_name, tint_id, tint_name, _flag = overprint
    v = item.attrs.get(color_id)
    if v is not None and v.as_u32() is not None:
        ink = carries_ink(w, v.as_u32())
    else:
        r = applied.item_value(color_name) if applied is not None else None
        ink = r is not None and reference_carries_ink(w, r)
    v = item.attrs.get(tint_id)
    if v is not None:
        tint = v.as_f64()
    else:
        t = applied.item_value(tint_name) if applied is not None else None
        try:
            tint = float(t) if t is not None else None
        except ValueError:
            tint = None
    return ink and tint != 0.0


def _item_overprint(w, x, item, applied, forced):
    """``OverprintFill`` and ``OverprintStroke`` of a page item, where
    its fill or stroke puts ink on the page."""
    for name, _c, _cn, _t, _tn, flag in OVERPRINT:
        overprint = next(o for o in OVERPRINT if o[0] == name)
        if not _puts_ink(w, item, applied, overprint):
            continue
        style = applied.item_value(name) if applied is not None else None
        v = item.attrs.get(flag)
        if v is not None:
            u = v.as_u32()
            if u is None or u > 1:
                continue
            value = str(u == 1).lower()
        elif forced(name):
            value = "false"
        else:
            continue
        if forced(name) or style != value:
            x.attr(name, value)


def _form_attrs(x, f):
    """The attributes of a form field that its attribute list holds."""
    if f.name is not None:
        x.attr("Name", f.name)
    attrs = [
        ("Description", f.description),
        ("ExportValue", f.export_value),
        ("Required", str(f.required).lower() if f.required is not None
         and f.kind not in ("button", "multi_state") else None),
        ("Multiline", str(f.multiline).lower() if f.multiline is not None else None),
        ("Scrollable", str(f.scrollable).lower() if f.scrollable is not None else None),
        ("FontSize", num(f.font_size) if f.font_size is not None else None),
        ("AppliedFont", f.font[0] if f.font is not None else None),
    ]
    for name, v in attrs:
        if v is not None:
            x.attr(name, v)
    if f.font is not None and f.font[1] is not None:
        x.attr("FontStyle", f.font[1])


def _form_states(w, x, item, f, outer):
    """The ``State`` elements of a form field with the items each
    shows. ``outer`` is the field's transform to the spread."""
    for s in f.states:
        if f.kind == "multi_state":
            statetype = ("long", str(s.id))
        else:
            t = _state_type(s.id)
            if t is None:
                w.warn(f"item {item.uid}: state type {s.id} is not known; "
                       f"state left out")
                continue
            statetype = ("enumeration", t)
        x.start("State") \
            .attr("Self", f"{uref(item.uid)}i{s.id:x}") \
            .attr("Active", str(s.active).lower()) \
            .attr("Enabled", str(s.enabled).lower())
        x.start("Properties") \
            .start("Statetype") \
            .attr("type", statetype[0]) \
            .text(statetype[1]) \
            .end() \
            .end()
        for i in s.items:
            page_item(w, x, i, True, outer)
        x.end()


def _form_behaviors(w, x, item, f):
    """The behaviours of a button, after its states, in stored order.
    Only buttons have behaviours in the samples; those of other fields
    are left out with a warning."""
    if not f.behaviors:
        return
    if f.kind != "button":
        w.warn(f"item {item.uid}: behaviours of a {f.kind} left out")
        return
    for b in f.behaviors:
        _behavior(w, x, b)


def _behavior(w, x, b):
    from .idml_animation import written_media

    def warn(what, code):
        w.warn(f"behaviour {b.uid}: {what} code {code} is not known; "
               f"left out")

    def zoom(code):
        if code == 0:
            return "InheritZoom"
        if code == 2:
            return "FitWindow"
        warn("zoom", code)
        return None

    def operation(code, codes):
        op = codes[code] if 0 <= code < len(codes) else None
        if op is None:
            warn("operation", code)
        return op

    def refs(show, fields):
        return " ".join(uref(u) for u, s in fields if s == show)

    def media(uid):
        return uref(uid) if uid in written_media(w) else None

    x.start(b.kind.element()).attr("Self", uref(b.uid))
    # Constants without a stored field: `AnchorName`,
    # `GoBackOnRollOff`, `LoopsToNextOrPrevious`.
    properties = None
    kind = b.kind.kind
    if kind[0] in ("first_page", "next_page", "previous_page", "next_view"):
        x.attr_opt("ZoomSetting", zoom(kind[1]))
    elif kind[0] == "page":
        x.attr_opt("ZoomSetting", zoom(kind[2])) \
            .attr("PageNumber", str(kind[1] + 1))
    elif kind[0] == "anchor":
        _file, anchor, z = kind[1], kind[2], kind[3]
        x.attr("AnchorName", "$ID/") \
            .attr_opt("ZoomSetting", zoom(z)) \
            .attr("FilePath", _file)
        item = 0 if not _file else anchor
        properties = ("AnchorItem", str(item))
    elif kind[0] in ("url", "submit_form"):
        x.attr_opt("URL", kind[1])
    elif kind[0] == "open_file":
        x.attr("FilePath", kind[1])
    elif kind[0] in ("clear_form", "print_form"):
        pass
    elif kind[0] == "show_hide":
        x.attr("FieldsToHide", refs(False, kind[1])) \
            .attr("FieldsToShow", refs(True, kind[1]))
    elif kind[0] == "state":
        x.attr("AssociatedMultiStateObject", uref(kind[1])) \
            .attr("GoBackOnRollOff", "false")
        properties = ("Statetype", str(kind[2]))
    elif kind[0] in ("next_state", "previous_state"):
        x.attr("AssociatedMultiStateObject", uref(kind[1])) \
            .attr("LoopsToNextOrPrevious", "true")
    elif kind[0] == "movie":
        x.attr_opt("MovieItem", media(kind[1])) \
            .attr_opt("NavigationPointID",
                      str(kind[3]) if kind[3] is not None else None) \
            .attr_opt("Operation", operation(kind[2], ["Play"]))
    elif kind[0] == "sound":
        x.attr_opt("SoundItem", media(kind[1])) \
            .attr_opt("Operation", operation(kind[2], ["Play", "Stop"]))
    if b.name.builtin:
        from .idml import Edition
        saving = w.saving_edition()
        name = Edition.behavior_name(saving, b.name.name) \
            if saving is not None else None
    else:
        name = b.name.name
    x.attr_opt("Name", name) \
        .attr("EnableBehavior", str(b.enabled).lower())
    event = {0x15602: "MouseDown", 0x15603: "MouseUp",
             0x15604: "MouseEnter"}.get(b.event)
    if event is None:
        warn("event", b.event)
    x.attr_opt("BehaviorEvent", event)
    if properties is not None or b.label:
        x.start("Properties")
        if properties is not None:
            tag, value = properties
            x.start(tag).attr("type", "long").text(value).end()
        label(x, b.label)
        x.end()
    x.end()


def frame_fitting(w, x, item):
    """``FrameFittingOption`` of a rectangle, oval or polygon. When
    the object style, the root ``[None]`` included, turns the fitting
    category off, IDML writes all seven values: the item's where it
    stores them, else the style's. Otherwise it writes the item's
    values that differ from the style's; a style other than the root
    without a category list (older files) gives all its values when
    none differ."""
    from .idml_attrs import FITTING_ATTRS

    style = w.doc.object_styles.get(item.object_style) \
        if item.object_style is not None else None
    all_ids = [attr_id for attr_id, _n, _k in FITTING_ATTRS]
    off = style is not None and style.enabled is not None \
        and 0x1B960 not in style.enabled
    if style is not None and off:
        from .attrs import Attrs
        merged = Attrs()
        merged.values = [(i, v) for i, v in style.fitting.values] \
            if style.fitting is not None else []
        merged.values = [(i, v) for i, v in merged.values
                         if item.attrs.get(i) is None]
        merged.values.extend((i, v) for i, v in item.attrs.values
                             if i in all_ids)
        attrs = fitting_attrs(w, merged, all_ids)
    else:
        differ = [attr_id for attr_id in all_ids
                  if item.attrs.get(attr_id) is not None
                  and (style is None or style.fitting is None
                       or style.fitting.get(attr_id)
                       != item.attrs.get(attr_id))]
        if differ:
            attrs = fitting_attrs(w, item.attrs, differ)
        elif style is not None and style.enabled is None \
                and not is_root_object_style(style):
            attrs = fitting_attrs(w, style.fitting, all_ids) \
                if style.fitting is not None else []
        else:
            attrs = []
    if attrs:
        x.empty("FrameFittingOption", attrs)


def _guide_origin(w, s, page, g, own_page, origin):
    """The point a guide's ``Location`` is measured from, by the ruler
    origin: the top-left corner of the spread's pages (``origin``);
    with ``PageOrigin`` the corner of the guide's own page; with
    ``SpineOrigin`` a vertical guide on a spread of two pages is
    measured from the left edge of the right page."""
    ruler = w.doc.prefs.ruler_origin
    if ruler == "page" and own_page:
        return spread_origin([page])
    if ruler == "spine" and not g.horizontal and len(s.pages) == 2:
        def left(q):
            return spread_origin([q])[0]

        right = max(left(s.pages[0]), left(s.pages[1]))
        return (right, origin[1])
    return origin


def guide(x, g, origin, shift, page_index):
    """A ruler guide. ``origin`` is the point IDML measures
    ``Location`` from and ``shift`` the spread's translation;
    ``page_index`` is IDML ``PageIndex``."""
    # Both points are taken to pasteboard coordinates before the
    # difference, which reproduces IDML's rounding (a location of 0
    # rather than 1e-13).
    if g.horizontal:
        from_v, by = origin[1], shift[1]
    else:
        from_v, by = origin[0], shift[0]
    x.start("Guide") \
        .attr("Self", uref(g.uid)) \
        .attr("Orientation", "Horizontal" if g.horizontal else "Vertical") \
        .attr("Location", num((g.position + by) - (from_v + by))) \
        .attr("FitToPage", str(g.fit_to_page).lower())
    # The only stored value in the samples, with the only IDML value.
    if g.view_threshold == 0.05:
        x.attr("ViewThreshold", "5")
    if g.layer != 0:
        x.attr("ItemLayer", uref(g.layer))
    if g.guide_type == 0:
        x.attr("GuideType", "Ruler")
    elif g.guide_type == 1:
        x.attr("GuideType", "Liquid")
    x.attr("Locked", str(g.locked).lower()) \
        .attr("PageIndex", str(page_index))
    if g.zone is not None:
        x.attr("GuideZone", num(g.zone))
    if g.overridden is not None:
        master, ids = g.overridden
        x.attr("OverriddenPageItemProps",
               "" if master == 0 else " ".join(str(i) for i in ids))
    elif g.zone is not None:
        x.attr("OverriddenPageItemProps", "")
    if g.color == 6:
        write_properties(x, [("GuideColor", "enumeration", ("text", "Cyan"))])
    x.end()


def page_settings(w, x, p, section):
    """The settings of a page: its tab order, overridden master items,
    layout grid use, layout rule and colour, and the values every IDML
    of the version has."""
    st = p.settings
    x.attr("TabOrder", " ".join(uref(u) for u in st.tab_order))
    overrides = []
    for item, with_ in st.overrides:
        overrides.append(uref(item))
        overrides.append(uref(with_ if with_ != 0 else None))
    x.attr("OverrideList", " ".join(overrides))
    # Codes 0 and 4 only; the others are in no pair.
    if st.grid_start == 0:
        x.attr("GridStartingPoint", "TopOutside")
    elif st.grid_start == 4:
        x.attr("GridStartingPoint", "CenterVertical")
    if st.use_master_grid is not None:
        x.attr("UseMasterGrid", str(st.use_master_grid).lower())
    if w.doc.version.major >= 8:
        rules = ["", "Recenter", "ObjectBased", "Scale", "GuideBased",
                 "UseMaster"]
        if st.layout_rule is None:
            x.attr("LayoutRule", "Off")
        elif 0 <= st.layout_rule < len(rules) and rules[st.layout_rule]:
            x.attr("LayoutRule", rules[st.layout_rule])
    x.attrs_missing(w.observed("Page"))
    props = []
    color = page_color(st.color)
    if color is not None:
        props.append(color)
    # A document page describes its numbering: section prefix, style,
    # continue, include prefix, page number (from DOM 20 twice) and
    # marker.
    if section is not None:
        s, number = section
        style = number_style(s.style)
        if style is not None:
            def item(ty, text):
                return Node(tag="ListItem", attrs=[("type", ty)], text=text)

            items = [
                item("string", s.prefix),
                # IDML writes the prefix again in place of this style.
                item("string", s.prefix) if s.style == Numbering.SINGLE_LEADING_ZEROS
                else item("enumeration", style),
                item("boolean", str(s.continue_numbering).lower()),
                item("boolean", str(s.include_prefix).lower()),
                item("long", str(number)),
            ]
            if w.doc.version.major >= 20:
                items.append(item("long", str(number)))
            # A return in the marker is a paragraph separator here.
            items.append(item("string", s.marker.replace("\r", "\u2029")))
            props.append(Node(tag="Descriptor",
                              attrs=[("type", "list")], children=items))
    write_properties(x, [], props)


def page_layout(w, x, p):
    """``MarginPreference`` and ``GridDataInformation`` of a page.
    Margins and columns are those in effect. The grid values that are
    the same in every sample are written only when the stored value is
    that one."""
    if p.margins is not None and p.columns is not None:
        m, c = p.margins, p.columns
        attrs = [
            ("ColumnCount", str(len(c.positions) // 2)),
            ("ColumnGutter", num(c.gutter)),
            ("Top", num(m.top)),
            ("Bottom", num(m.bottom)),
            ("Left", num(m.left)),
            ("Right", num(m.right)),
        ]
        # The last u16 of the columns chunk.
        if c.direction == 0:
            attrs.append(("ColumnDirection", "Horizontal"))
        elif c.direction == 1:
            attrs.append(("ColumnDirection", "Vertical"))
        attrs.append(("ColumnsPositions", nums(c.positions)))
        x.empty("MarginPreference", attrs)
    if p.grid is not None:
        grid_data(w, x, p.grid)


def grid_data(w, x, g):
    """``GridDataInformation`` of layout grid settings (chunk
    0xCD02)."""
    x.start("GridDataInformation") \
        .attr("FontStyle", g.font_style)
    # `PointSize`, `CharacterAki` and `LineAki` are located; the
    # scales are written only when they have the value of every sample.
    size, character_aki, line_aki, h_scale, v_scale = g.numbers
    for name, value, observed, scale in [
            ("PointSize", size, None, 1.0),
            ("CharacterAki", character_aki, None, 1.0),
            ("LineAki", line_aki, None, 1.0),
            ("HorizontalScale", h_scale, 1.0, 100.0),
            ("VerticalScale", v_scale, 1.0, 100.0),
    ]:
        if observed is None or value == observed:
            x.attr(name, num(value * scale))
    if g.codes == [3, 0, 3, 1]:
        x.attr("LineAlignment", "LeftOrTopLineJustify") \
            .attr("GridAlignment", "AlignEmCenter") \
            .attr("CharacterAlignment", "AlignEmCenter")
    f = w.doc.fonts.get(g.font)
    if f is not None:
        write_properties(x, [("AppliedFont", "string",
                              ("text", w.family_names(f)[0]))])
    x.end()


def spread(w, s, master, page_index, names):
    """A spread or master spread. ``page_index`` counts the document
    pages written so far (a one-element list); ``names`` gives each
    document page its name."""
    from .idml_animation import timing_setting

    x = Xml()
    kind = "MasterSpread" if master else "Spread"
    w.package_root(x, kind)
    x.start(kind) \
        .attr("Self", uref(s.uid)) \
        .attr("PageCount", str(len(s.pages)))
    origin = spread_origin(s.pages)
    major = w.doc.version.major
    if not master:
        x.attr("BindingLocation", str(s.binding_location))
    x.attr("ItemTransform", matrix(s.transform))
    if s.master_name is not None:
        prefix, base = s.master_name
    else:
        prefix, base = "A", "Master"
    if master:
        x.attr("Name", f"{prefix}-{base}") \
            .attr("NamePrefix", prefix) \
            .attr("BaseName", base)
        if s.show_master_items is None:
            x.attr("ShowMasterItems", "true")
        elif s.show_master_items == 0:
            x.attr("ShowMasterItems", "false")
    else:
        # Page transitions; without the chunk, none.
        ty, direction = s.transition if s.transition is not None else (0, 0)
        t = next((n for c, n in TRANSITION_TYPES if c == ty), None)
        if t is not None:
            x.attr("PageTransitionType", t)
        d = next((n for c, n in TRANSITION_DIRECTIONS if c == direction), None)
        if d is not None:
            x.attr("PageTransitionDirection", d)
        if s.shuffle == 0:
            x.attr("AllowPageShuffle", "true")
        elif s.shuffle in (1, None):
            x.attr("AllowPageShuffle", "false")
    x.attrs_missing(w.observed(kind))
    if master and major >= 8:
        f = _primary_text_frame(s)
        if f is not None:
            x.attr("PrimaryTextFrame", f)
    # A master spread has the colour its pages share.
    if master and s.pages \
            and all(p.settings.color == s.pages[0].settings.color
                    for p in s.pages):
        n = page_color(s.pages[0].settings.color)
        if n is not None:
            x.start("Properties")
            n.write(x)
            x.end()
    if not master:
        fp = element("Spread/FlattenerPreference", major)
        if fp is not None:
            # Without the flattener chunk, IDML has 400 and 400 after a
            # session of a Japanese or Chinese edition, otherwise 300
            # and 150.
            res = s.flattener_resolution
            if res is None:
                japanese = w.japanese_session()
                if japanese is True:
                    res = (400.0, 400.0)
                elif japanese is False:
                    res = (300.0, 150.0)
            if res is not None:
                line_art, gradient = res
                fp.attrs.insert(0, ("GradientAndMeshResolution", num(gradient)))
                fp.attrs.insert(0, ("LineArtAndTextResolution", num(line_art)))
            fp.write(x)
    timing_setting(w, x, uref(s.uid), s.timing)
    for p in s.pages:
        if master:
            name = prefix
        else:
            page_index[0] += 1
            name = names[page_index[0] - 1] if page_index[0] - 1 < len(names) \
                else str(page_index[0])
        # Master pages have none (`n`).
        layout = None if master else \
            (w.page_layouts[page_index[0] - 1]
             if page_index[0] - 1 < len(w.page_layouts) else None)
        x0, y0, x1, y1 = p.bounds
        x.start("Page") \
            .attr("Self", uref(p.uid)) \
            .attr("Name", name) \
            .attr("GeometricBounds", nums([y0, x0, y1, x1])) \
            .attr("ItemTransform", matrix(p.transform)) \
            .attr("AppliedMaster", uref(p.master)) \
            .attr("MasterPageTransform", matrix(p.master_transform))
        if w.doc.version.major >= 8:
            x.attr("AppliedAlternateLayout", uref(layout))
        section = None if master else \
            (w.page_sections[page_index[0] - 1]
             if page_index[0] - 1 < len(w.page_sections) else None)
        page_settings(w, x, p, section)
        # A guide belongs to a page, or to the spread; IDML writes a
        # spread's guides in its first page.
        first = s.pages and p.uid == s.pages[0].uid
        # Pages count from the spine: left of it -1, -2, ..., right of
        # it 1, 2, ...; a master spread's spine is at its left edge.
        position = next((i for i, q in enumerate(s.pages) if q.uid == p.uid), 0)
        binding = 0 if master else s.binding_location
        from_spine = position - binding if position < binding \
            else position - binding + 1
        extent = _pages_extent(s.pages)
        shift = (s.transform.values[4], s.transform.values[5])
        for g in s.guides:
            own_page = any(q.uid == g.owner for q in s.pages)
            # IDML leaves out a guide of the spread that lies outside
            # the spread's pages.
            outside = False
            if not own_page and extent is not None:
                lo, hi = (extent[1], extent[3]) if g.horizontal \
                    else (extent[0], extent[2])
                outside = g.position < lo - 1e-6 or g.position > hi + 1e-6
            if outside:
                continue
            if g.owner == p.uid or (first and not own_page):
                index = from_spine if own_page else 0
                origin_g = _guide_origin(w, s, p, g, own_page, origin)
                guide(x, g, origin_g, shift, index)
        page_layout(w, x, p)
        x.end()
    for item in s.items:
        page_item(w, x, item, False, s.transform)
    x.end()
    return x.finish()


def _primary_text_frame(s):
    """``PrimaryTextFrame`` of a master spread: ``n`` without a primary
    story, otherwise the spread's first text frame of that story;
    ``None`` if the spread has no such frame."""
    def find(items, story):
        for i in items:
            if i.kind[0] == "text_frame" \
                    and i.kind[1].get("story") == story \
                    and i.kind[1].get("previous") is None:
                return i.uid
            found = find(i.children, story)
            if found is not None:
                return found
        return None

    if s.primary_story is None or s.primary_story == 0:
        return "n"
    found = find(s.items, s.primary_story)
    return uref(found) if found is not None else None
