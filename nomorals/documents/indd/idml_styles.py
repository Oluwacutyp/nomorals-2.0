"""``Resources/Styles.xml``: paragraph, character, object, table and
cell styles, their style groups, and the root styles every IDML has.

A faithful port of ``src/idml/styles.rs`` from storytold/indd-utils
(commit 136f973, dual MIT/Apache-2.0 — see attribution.py).
"""

from __future__ import annotations

from .idml_attrs import (
    CELL_STYLE_PARAGRAPH_STYLE,
    CONTOUR_PARAMS,
    FITTING_ATTRS,
    anchored_settings,
    contour_type,
    fitting_attrs,
    item_attr_values,
    text_attrs_of,
    text_wrap_mode,
    text_wrap_offsets,
)
from .idml_designmap import named_grid_ref, toc_styles
from .idml_export import object_export_option
from .idml_spread import BASELINE_RELATIVE, FIRST_BASELINE, JUSTIFY, POINTS, SIZING
from .idml_values import Node, list_values
from .idml_values import element as values_element
from .idml_values import object_style as values_object_style
from .idml_values import root_style as values_root_style
from .idml_values import when_written as values_when_written
from .idml_xml import (
    Xml,
    builtin_key,
    num,
    self_name,
    set_property,
    style_name,
    ui_color_property,
    uref,
    write_properties,
)

#: The attribute of a table style's caption position (0x1047D).
TABLE_STYLE_CAPTION_POSITION = 0x1047D


def style_shortcut(s, platform):
    """``KeyboardShortcut`` and ``ExtendedKeyboardShortcut`` of a style,
    from its stored key. ``None`` where the samples do not show the
    value."""
    if s.shortcut is None:
        return None, None
    key, low, high = s.shortcut
    if key == 0:
        return "0 0", "0 0 0"
    ch = key & 0xFFFF
    mods = low + 256 * high
    if key >> 16 == 0xC000 and 0x30 <= ch <= 0x39:
        d = ch - 0x30
        # The key code follows the platform of the last session.
        if platform == 3:
            code = 96 + d
        elif platform == 0:
            code = 82 + d + (1 if d >= 8 else 0)
        else:
            code = None
        return (f"{mods} {code}" if code is not None else None, "0 0 0")
    if key >> 16 == 0x8000:
        return "0 0", f"{mods} {ch} 1"
    return None, None


def group_paths(doc):
    """The path of names to each style and style group, by UID."""
    out = {}

    def walk(g, path):
        for c in g.children:
            out[c] = list(path)
            sub = doc.style_groups.get(c)
            if sub is not None:
                p = list(path) + [sub.name]
                out[c] = p
                walk(sub, p)

    for g in doc.style_groups.values():
        if g.root is not None:
            walk(g, [])
    return out


def root_lists(props, major):
    """Leaves out the list property that IDML does not write on the
    root paragraph style and on ``TextDefault``: an empty ``TabList``
    before DOM 8."""
    props[:] = [p for p in props
                if not (p[2][0] == "list" and not p[2][1]
                        and p[0] == "TabList" and major < 8)]


def root_values(w, tag, written, props):
    """Observed values of a root style that are not in ``written``
    (attributes) or ``props`` (Properties children): attributes,
    Properties children and other child elements."""
    node = values_root_style(tag, w.doc.version.major)
    attrs = [(k, v) for k, v in node.attrs if k not in written]
    extra_props = []
    children = []
    for c in node.children:
        if c.tag == "Properties":
            extra_props.extend(
                p for p in c.children
                if not any(n == p.tag for n, _t, _v in props))
        else:
            children.append(c)
    return attrs, extra_props, children


def table_style_element(w, x, tag, s, styles):
    """A cell or table style other than the root style."""
    from .idml_attrs import cell_attrs, cell_style_edge_attrs, table_attrs

    x.start(tag) \
        .attr("Self", table_style_ref(w, tag, s)) \
        .attr("Name", table_style_name(w, s))
    if tag == "CellStyle":
        attrs = cell_attrs(w, s.attrs)
        # The IDML schema has no `WritingDirection` on cell styles.
        attrs = [(n, v) for n, v in attrs if n != "WritingDirection"]
        attrs.extend(cell_style_edge_attrs(w, s.attrs))
    else:
        attrs = table_attrs(w, s.attrs, True)
    for k, v in attrs:
        x.attr(k, v)
    if tag == "TableStyle":
        caption_position(w, x, s.attrs)
    # Values every IDML has on these styles.
    n = values_element(tag, w.doc.version.major)
    if n is not None:
        x.attrs_missing(n.attrs)
    if tag == "CellStyle":
        # A style without its own paragraph style has none, whatever
        # its based-on style has.
        p = s.attrs.get(CELL_STYLE_PARAGRAPH_STYLE)
        p = p.as_u32() if p is not None else None
        if p is not None and p in w.doc.styles:
            x.attr("AppliedParagraphStyle", w.style_ref(p, True))
        elif p is None:
            x.attr("AppliedParagraphStyle",
                   "ParagraphStyle/$ID/[No paragraph style]")
    if s.based_on is not None:
        base = styles.get(s.based_on)
        if base is not None:
            # A root style is written as a string.
            if base.builtin and base.based_on is None:
                prop = ("BasedOn", "string", ("text", table_style_name(w, base)))
            else:
                prop = ("BasedOn", "object",
                        ("text", table_style_ref(w, tag, base)))
            write_properties(x, [prop])
    x.end()


def root_table_style(w, x, tag, name):
    """A root cell or table style. The root table style takes the
    values of its attribute list, and from DOM 11 its text and graphic
    cell values."""
    from .idml_attrs import GRAPHIC_CELL_ATTRS, TEXT_CELL_ATTRS, table_attrs

    attrs, props, children = root_values(w, tag, [], [])
    x.start(tag) \
        .attr("Self", f"{tag}/$ID/{name}") \
        .attr("Name", builtin_key(name))
    for k, v in attrs:
        x.attr(k, v)
    root = next((s for s in w.doc.table_styles.values()
                 if s.builtin and s.based_on is None and s.name == name),
                None)
    if tag == "TableStyle" and root is not None:
        for k, v in table_attrs(w, root.attrs, True):
            x.attr(k, v)
        caption_position(w, x, root.attrs)
        if w.doc.version.major >= 11:
            for attr_id, k, kind in list(TEXT_CELL_ATTRS) \
                    + list(GRAPHIC_CELL_ATTRS):
                v = root.attrs.get(attr_id)
                if v is not None:
                    from .idml_attrs import value_text
                    text = value_text(w, kind, v)
                    if text is not None:
                        x.attr(k, text)
    write_properties(x, [], props)
    for c in children:
        c.write(x)
    x.end()


def caption_position(w, x, attrs):
    """``CaptionPosition`` of a table style from DOM 21.5, from its own
    attribute 0x1047D only."""
    if w.dom_version() < (21, 5):
        return
    v = attrs.get(TABLE_STYLE_CAPTION_POSITION)
    if v is None:
        position = "AfterTable"
    elif v.kind == "enum" and v.data == 1:
        position = "BeforeTable"
    else:
        return
    x.attr("CaptionPosition", position)


def table_style_ref(w, tag, s):
    """``CellStyle/...`` or ``TableStyle/...`` reference of a style."""
    return f"{tag}/{self_name(table_style_name(w, s))}"


def table_style_name(w, s):
    """The IDML name of a table or cell style, with its group path."""
    return w.grouped_name(s.uid, s.name, s.builtin)


def cell_style_ref(w, uid):
    """The reference for an applied cell style UID; 0 is ``[None]``."""
    if uid == 0:
        return "CellStyle/$ID/[None]"
    s = w.doc.cell_styles.get(uid)
    if s is None:
        return None
    return table_style_ref(w, "CellStyle", s)


def styles(w):
    """``Resources/Styles.xml``."""
    from .idml_applied import is_root_object_style
    from .idml_transparency import style_settings as transparency_settings
    from .ids import RootKind

    doc = w.doc
    x = Xml()
    w.package_root(x, "Styles")

    def root_of(kind):
        return next((g for g in doc.style_groups.values()
                     if g.root == kind), None)

    for paragraph in (False, True):
        if paragraph:
            group_tag, tag, sub_tag = ("RootParagraphStyleGroup",
                                       "ParagraphStyle",
                                       "ParagraphStyleGroup")
            kind = RootKind.PARAGRAPH
        else:
            group_tag, tag, sub_tag = ("RootCharacterStyleGroup",
                                       "CharacterStyle",
                                       "CharacterStyleGroup")
            kind = RootKind.CHARACTER
        root = root_of(kind)
        x.start(group_tag).attr(
            "Self", uref(root.uid) if root is not None else group_tag)
        written = set()
        if root is not None:
            style_group_children(w, x, root, tag, sub_tag, written)
        # Styles not listed in a group (such as a built-in root style).
        # Some documents have two style objects with the same name,
        # which would give two elements with the same `Self`; the first
        # is written.
        refs = {w.style_ref(u, paragraph) for u in written}
        for s in sorted((s for s in doc.styles.values()
                         if s.paragraph == paragraph),
                        key=lambda s: s.uid):
            if s.uid not in written:
                ref = w.style_ref(s.uid, paragraph)
                if ref not in refs:
                    refs.add(ref)
                    style_element(w, x, s, tag)
        x.end()
    toc_styles(w, x)
    for tag, kind, style, name in [
            ("RootCellStyleGroup", RootKind.CELL, "CellStyle", "[None]"),
            ("RootTableStyleGroup", RootKind.TABLE, "TableStyle",
             "[No table style]"),
    ]:
        root = root_of(kind)
        ident = uref(root.uid) if root is not None else tag
        x.start(tag).attr("Self", ident)
        root_table_style(w, x, style, name)
        styles_map = doc.cell_styles if style == "CellStyle" \
            else doc.table_styles
        # Styles and style groups listed in the root group first, then
        # any other styles; the root style is written above.
        seen = set()
        r = next((s for s in styles_map.values()
                  if s.builtin and s.based_on is None and s.name == name),
                 None)
        if r is not None:
            seen.add(table_style_ref(w, style, r))
        if root is not None:
            table_style_group_children(w, x, root, style, styles_map,
                                       seen)
        for s in sorted(styles_map.values(), key=lambda s: s.uid):
            if table_style_ref(w, style, s) not in seen:
                seen.add(table_style_ref(w, style, s))
                table_style_element(w, x, style, s, styles_map)
        x.end()
    object_root = root_of(RootKind.OBJECT)
    x.start("RootObjectStyleGroup").attr(
        "Self",
        uref(object_root.uid) if object_root is not None
        else "RootObjectStyleGroup")
    for os in sorted(doc.object_styles.values(), key=lambda s: s.uid):
        name = w.grouped_name(os.uid, os.name, os.builtin)
        root = os.builtin and os.name == "[None]" and os.based_on is None
        node = object_style_node(w, os, root)
        if not root:
            r = next((s for s in doc.object_styles.values()
                      if is_root_object_style(s)), None)
            if r is not None:
                settings, left_out = transparency_settings(w, os, r)
                # Before the effects categories, as IDML has them.
                at = next((i for i, c in enumerate(node.children)
                           if c.tag
                           == "ObjectStyleObjectEffectsCategorySettings"),
                          len(node.children))
                node.children[at:at] = settings
                for effect, attr, v in left_out:
                    w.warn(f"object style {os.uid}: {effect} {attr} {v} "
                           f"is outside the IDML range; left out")
        x.start("ObjectStyle") \
            .attr("Self", f"ObjectStyle/{self_name(name)}") \
            .attr("Name", name)
        for k, v in node.attrs:
            x.attr(k, v)
        # The values of every object style from DOM 21.2.
        if w.dom_version() >= (21, 2):
            x.attr("EpubAriaLabel", "") \
                .attr("EpubAriaLabelSourceType", "NoneARIALabel")
        props = []
        props_node = next((c for c in node.children
                           if c.tag == "Properties"), None)
        if props_node is not None:
            props = list(props_node.children)
        if not root and os.based_on is not None:
            base = doc.object_styles.get(os.based_on)
            if base is not None:
                # The root "[None]" is written as a string.
                if base.builtin and base.name == "[None]":
                    ty, text = "string", builtin_key(base.name)
                else:
                    ty, text = "object", \
                        f"ObjectStyle/{self_name(w.grouped_name(base.uid, base.name, base.builtin))}"
                props.insert(0, Node(tag="BasedOn",
                                     attrs=[("type", ty)], text=text))
        if props:
            x.start("Properties")
            for p in props:
                p.write(x)
            x.end()
        for c in node.children:
            if c.tag != "Properties":
                c.write(x)
        x.end()
    x.end()
    # The trap presets every IDML has.
    for n in list_values("TrapPreset", doc.version.major):
        n.write(x)
    x.end()
    return x.finish()


def carries_ink(w, uid):
    """Whether swatch ``uid`` puts ink on the page: a swatch other
    than ``None`` and ``Paper``, and not a white process colour (CMYK
    0 0 0 0 or RGB 255 255 255)."""
    r = w.doc.swatches.get(uid)
    if r is None:
        return False
    if r in ("Swatch/None", "Color/Paper"):
        return False
    for c in w.doc.colors:
        if c.uid == uid and c.model == 0:
            if c.space == ("cmyk",) \
                    and all(v == 0.0 for v in c.values):
                return False
            if c.space == ("rgb",) and all(v == 1.0 for v in c.values):
                return False
    return True


def reference_carries_ink(w, r):
    """Whether the swatch with IDML reference ``r`` puts ink on the
    page (``carries_ink``)."""
    for uid, s in w.doc.swatches.items():
        if s == r:
            return carries_ink(w, uid)
    return False


def overprint_attrs(w, attrs, root):
    """``OverprintFill``, ``OverprintStroke`` and ``OverprintGap`` of
    an object style or the page item defaults: where the colour
    carries ink (never on the root style), and from 21.4 always."""
    out = []
    for name, color, tint, overprint in [
            ("OverprintFill", 0x6E68, 0x6E69, 0x6E6A),
            ("OverprintStroke", 0x6E64, 0x6E66, 0x6E67),
            ("OverprintGap", 0x6E89, 0x6E8A, 0x6E8B),
    ]:
        v = attrs.get(overprint)
        v = v.as_u32() if v is not None else None
        if v is None:
            continue

        def ink(tint=tint, color=color):
            tint_v = attrs.get(tint)
            tint_v = tint_v.as_f64() if tint_v is not None else None
            color_v = attrs.get(color)
            color_v = color_v.as_u32() if color_v is not None else None
            return color_v is not None and carries_ink(w, color_v) \
                and tint_v != 0.0

        if v <= 1 and (w.saved_by((21, 4)) or (not root and ink())):
            out.append((name, str(v == 1).lower()))
    return out


def object_style_node(w, os, root):
    """An object style as a values node: the values every exported
    IDML has on such a style, with the values read from the INDD in
    their place."""
    from .model import GeneratedAltText

    major = w.doc.version.major
    version = (major, w.doc.version.minor)
    node = values_root_style("ObjectStyle", major) if root \
        else values_object_style(major)
    # Every object style has the preferences' baseline frame grid
    # colour.
    c = w.doc.prefs.baseline_frame_grid_color
    if c is not None:
        from .idml_xml import frame_grid_color
        colour = frame_grid_color(c)
        if colour is not None:
            b = next((c for c in node.children
                      if c.tag == "BaselineFrameGridOption"), None)
            if b is not None:
                set_property(b, colour)
    # The export options of the version, with the style's alternative
    # text and tagging, in place of those of the value files.
    node.children = [c for c in node.children
                     if c.tag != "ObjectExportOption"]
    e = object_export_option(w.doc.version, os.export_options, None,
                             GeneratedAltText(), "style")
    if e is not None:
        node.children.insert(0, e)
    # The transform values the style sets, where IDML has the element.
    transform = [(name, num(v)) for name, v in zip(
        ["TransformAttrX", "TransformAttrY", "TransformAttrHeight",
         "TransformAttrWidth"], os.transform, strict=True)
        if v is not None]
    if transform and node.child("TransformAttributeOption") is not None:
        node.set(["TransformAttributeOption"], transform)
    # Flex paddings and gaps; without the chunk 12 and 20.
    if node.child("FlexLayoutAttributeOption") is not None:
        v = os.flex_spacing if os.flex_spacing is not None \
            else [12.0, 12.0, 12.0, 12.0, 20.0, 20.0]
        names = ["FlexPaddingTop", "FlexPaddingRight",
                 "FlexPaddingBottom", "FlexPaddingLeft",
                 "FlexGapRow", "FlexGapColumn"]
        node.set(["FlexLayoutAttributeOption"],
                 [(n, num(vv)) for n, vv in zip(names, v, strict=True)])
    attrs = item_attr_values(w, os.attrs)
    for attr_id, name in [(0x551E, "GradientFillAngle"),
                          (0x5524, "GradientStrokeAngle")]:
        v = os.attrs.get(attr_id)
        if v is not None:
            f = v.as_f64()
            if f is not None:
                attrs.append((name, num(f)))
    g = named_grid_ref(w, os.named_grid)
    if g is not None:
        attrs.append(("AppliedNamedGrid", g))
    attrs.extend(overprint_attrs(w, os.attrs, root))
    for name, value in export_flags(os.export, w.saved_by, False):
        attrs.append((name, value))
    if os.paragraph_style is None or os.paragraph_style == 0:
        # No chunk 0x1B946 is no style too.
        attrs.append(("AppliedParagraphStyle", "n"))
    elif os.paragraph_style in w.doc.styles:
        attrs.append(("AppliedParagraphStyle",
                      w.style_ref(os.paragraph_style, True)))
    # Chunk 0x1B94D; without it, false. The root style has none.
    # A style without a keyboard shortcut; other keys are not decoded.
    if not root and os.shortcut_key == 0:
        attrs.append(("KeyboardShortcut", "0 0"))
    if not root:
        on = os.apply_next == 1
        if os.apply_next is None or os.apply_next in (0, 1):
            attrs.append(("ApplyNextParagraphStyle", str(on).lower()))
    effects = []
    # The root `[None]` has no `Enable…` attributes and no effects
    # category settings in IDML.
    if os.enabled is not None and not root:
        on = os.enabled
        # Each attribute is true when its category ID is in the list,
        # from the version on which IDML has it. Where several IDs are
        # present or absent together in every sample, the attribute is
        # written only when they agree.
        categories = [
            ([0x1B933], (7, 0), ["EnableFill"]),
            ([0x1B934], (7, 0), ["EnableStroke"]),
            ([0x1B935, 0x1B936], (7, 0),
             ["EnableStrokeAndCornerOptions"]),
            ([0x1B93E], (7, 0), ["EnableTextFrameGeneralOptions"]),
            ([0xADC8], (7, 0), ["EnableTextFrameBaselineOptions"]),
            ([0xADC9], (8, 0), ["EnableTextFrameAutoSizingOptions"]),
            ([0x1B940], (7, 0), ["EnableStoryOptions"]),
            ([0x1B960], (7, 0), ["EnableFrameFittingOptions"]),
            ([0x1B93F], (7, 0), ["EnableParagraphStyle"]),
            ([0xADCB], (15, 1), ["EnableTextFrameColumnRuleOptions"]),
            ([0xCA2F], (7, 0), ["EnableAnchoredObjectOptions"]),
            ([0x1B942, 0x37C8, 0x37C9], (7, 0),
             ["EnableTextWrapAndOthers"]),
            ([0xADCA], (12, 0), ["EnableTextFrameFootnoteOptions"]),
            ([0x6EA1, 0x6EA2, 0x6EA3, 0x6EA4, 0x6EA5, 0x6EA6, 0x6EA7],
             (13, 0), ["EnableTransformAttributes"]),
            ([0x1B97A, 0x1B97C, 0x1B97D, 0x1B97E], (9, 0),
             ["EnableExportTagging",
              "EnableObjectExportAltTextOptions",
              "EnableObjectExportEpubOptions",
              "EnableObjectExportTaggedPdfOptions"]),
        ]
        for ids, since, names in categories:
            first = ids[0] in on
            if version >= since \
                    and all((i in on) == first for i in ids):
                for name in names:
                    attrs.append((name, str(first).lower()))
        # The effects categories: transparency of the object, and of
        # its fill, stroke and content together.
        effects.append(("ObjectStyleObjectEffectsCategorySettings",
                        0x1B937 in on))
        parts = [0x1B948, 0x1B950, 0x1B958]
        first = parts[0] in on
        if all((i in on) == first for i in parts):
            for tag in ["ObjectStyleFillEffectsCategorySettings",
                        "ObjectStyleStrokeEffectsCategorySettings",
                        "ObjectStyleContentEffectsCategorySettings"]:
                effects.append((tag, first))
    node.set([], attrs)
    for tag, on in effects:
        node.set([tag], [("EnableTransparency", str(on).lower())])
    if os.frame is not None:
        fr = os.frame
        tf = []
        if fr.column_count is not None:
            tf.append(("TextColumnCount", str(fr.column_count)))
        if fr.column_gutter is not None:
            tf.append(("TextColumnGutter", num(fr.column_gutter)))
        if fr.column_fixed_width is not None:
            tf.append(("TextColumnFixedWidth", num(fr.column_fixed_width)))
        # Fields of the text frame settings that use the frame codes.
        if fr.vertical_balance_columns is not None:
            tf.append(("VerticalBalanceColumns",
                       str(fr.vertical_balance_columns).lower()))
        if fr.use_fixed_width is not None:
            tf.append(("UseFixedColumnWidth",
                       str(fr.use_fixed_width).lower()))
        if fr.vertical_justification is not None:
            v = JUSTIFY[fr.vertical_justification] \
                if 0 <= fr.vertical_justification < len(JUSTIFY) else None
            if v is not None:
                tf.append(("VerticalJustification", v))
        if fr.first_baseline_offset is not None:
            v = FIRST_BASELINE[fr.first_baseline_offset] \
                if 0 <= fr.first_baseline_offset < len(FIRST_BASELINE) \
                else None
            if v is not None:
                tf.append(("FirstBaselineOffset", v))
        if major >= 8 and fr.auto_sizing is not None:
            ty, point = fr.auto_sizing
            ty_name = SIZING[ty] if 0 <= ty < len(SIZING) else None
            point_name = POINTS[point] \
                if 0 <= point < len(POINTS) else None
            if ty_name is not None and point_name is not None:
                tf.append(("AutoSizingType", ty_name))
                tf.append(("AutoSizingReferencePoint", point_name))
        if major >= 8 and fr.no_line_breaks is not None:
            tf.append(("UseNoLineBreaksForAutoSizing",
                       str(fr.no_line_breaks).lower()))
        if major >= 8 and fr.minimum_sizes is not None:
            (use_height, use_width), (height, width) = fr.minimum_sizes
            tf.append(("UseMinimumHeightForAutoSizing",
                       str(use_height).lower()))
            tf.append(("MinimumHeightForAutoSizing", num(height)))
            tf.append(("UseMinimumWidthForAutoSizing",
                       str(use_width).lower()))
            tf.append(("MinimumWidthForAutoSizing", num(width)))
        footnote = []
        if fr.footnotes is not None:
            span, minimum, between = fr.footnotes
            if span <= 1:
                span_text = str(span == 1).lower()
                # In `TextFramePreference` from 13.1.
                if w.saved_by((13, 1)):
                    tf.append(("FootnotesSpanAcrossColumns", span_text))
                    tf.append(("FootnotesMinimumSpacing", num(minimum)))
                    tf.append(("FootnotesSpaceBetween", num(between)))
                footnote = [
                    ("SpanFootnotesAcross", span_text),
                    ("MinimumSpacingOption", num(minimum)),
                    ("SpaceBetweenFootnotes", num(between)),
                ]
        if fr.column_rule is not None:
            width, color, tint = fr.column_rule
            colour = "n" if color == 0 else w.doc.swatches.get(color)
            tf.append(("ColumnRuleStrokeWidth", num(width)))
            if colour is not None:
                tf.append(("ColumnRuleStrokeColor", colour))
            tf.append(("ColumnRuleStrokeTint", num(tint)))
        node.set(["TextFramePreference"], tf)
        # Only the top inset is shown apart from the others; the list
        # is written when all four are equal.
        if fr.insets is not None:
            a, b, c, e = fr.insets
            if a == b == c == e:
                props = node.set(["TextFramePreference", "Properties"], [])
                props.children = [ch for ch in props.children
                                  if ch.tag != "InsetSpacing"]
                props.children.append(Node(
                    tag="InsetSpacing", attrs=[("type", "list")],
                    children=[Node(tag="ListItem",
                                   attrs=[("type", "unit")], text=num(a))
                              for _ in range(4)]))
        # The style's baseline frame grid; the colour stays the
        # preferences'.
        if fr.baseline_grid is not None \
                and node.child("BaselineFrameGridOption") is not None:
            g = fr.baseline_grid
            attrs_g = [
                ("UseCustomBaselineFrameGrid",
                 str(g.use_custom).lower()),
                ("StartingOffsetForBaselineFrameGrid", num(g.start)),
            ]
            r = BASELINE_RELATIVE[g.relative] \
                if 0 <= g.relative < len(BASELINE_RELATIVE) else None
            if r is not None:
                attrs_g.append(("BaselineFrameGridRelativeOption", r))
            attrs_g.append(("BaselineFrameGridIncrement",
                            num(g.increment)))
            node.set(["BaselineFrameGridOption"], attrs_g)
        if footnote and node.child("TextFrameFootnoteOptionsObject") \
                is not None:
            node.set(["TextFrameFootnoteOptionsObject"], footnote)
    # Every object style has it from 13.1.
    if w.saved_by((13, 1)):
        node.merge(Node(tag="ObjectStyle", children=[Node(
            tag="TextFramePreference",
            attrs=[("FootnotesEnableOverrides", "false")])]))
    story = []
    if os.story is not None:
        st = os.story
        if st.frame_type == 0:
            story.append(("FrameType", "Unknown"))
        elif st.frame_type == 1:
            story.append(("FrameType", "TextFrameType"))
        elif st.frame_type == 2:
            story.append(("FrameType", "FrameGridType"))
        if st.orientation == 0:
            story.append(("StoryOrientation", "Unknown"))
        elif st.orientation == 1:
            story.append(("StoryOrientation", "Horizontal"))
        story.append(("OpticalMarginSize", num(st.optical_size)))
        if st.optical_alignment == 0:
            story.append(("OpticalMarginAlignment", "false"))
        elif st.optical_alignment == 1:
            story.append(("OpticalMarginAlignment", "true"))
    if os.direction == 1:
        story.append(("StoryDirection", "LeftToRightDirection"))
    elif os.direction is None or os.direction == 0:
        story.append(("StoryDirection", "UnknownDirection"))
    node.set(["StoryPreference"], story)
    # The u16 at 40 of the wrap chunk: 0 for
    # `ApplyToMasterPageOnly`; without the chunk, false.
    if node.child("TextWrapPreference") is not None:
        master_flag = os.text_wrap.master_flag \
            if os.text_wrap is not None else None
        if master_flag is None or master_flag == 1:
            node.set(["TextWrapPreference"],
                     [("ApplyToMasterPageOnly", "false")])
        elif master_flag == 0:
            node.set(["TextWrapPreference"],
                     [("ApplyToMasterPageOnly", "true")])
    mode = text_wrap_mode(os.text_wrap)
    if mode is not None:
        node.set(["TextWrapPreference"], [("TextWrapMode", mode)])
        node.set(["TextWrapPreference", "Properties", "TextWrapOffset"],
                 text_wrap_offsets(os.text_wrap))
        # A style without chunk 0x3777 has the default contour.
        c = os.contour if os.contour is not None else _default_contour()
        t = contour_type(c.kind)
        if t is not None:
            attrs_c = [("ContourType", t)]
            if c.inside_edges is not None:
                attrs_c.append(("IncludeInsideEdges",
                                str(c.inside_edges).lower()))
            attrs_c.append(("ContourPathName", "$ID/"))
            if w.dom_version() >= (21, 4):
                attrs_c.extend(CONTOUR_PARAMS)
            node.set(["TextWrapPreference", "ContourOption"], attrs_c)
    all_ids = [attr_id for attr_id, _n, _k in FITTING_ATTRS]
    node.set(["FrameFittingOption"],
             fitting_attrs(w, os.fitting, all_ids))
    if os.anchor is not None:
        node.set(["AnchoredObjectSetting"],
                 anchored_settings(os.anchor))
    return node


def _default_contour():
    """The default contour of a text wrap (chunk 0x3777 absent)."""
    from .model import Contour
    return Contour()


def table_style_group_children(w, x, g, tag, styles, seen):
    """The cell or table styles and style groups of a style group, in
    its order. ``seen`` holds the ``Self`` of the styles written: some
    documents have two style objects with the same name, and the first
    is written."""
    for c in g.children:
        sub = w.doc.style_groups.get(c)
        if sub is not None:
            sub_tag = f"{tag}Group"
            name = group_name(w, sub)
            x.start(sub_tag) \
                .attr("Self", f"{sub_tag}/$ID/{self_name(name)}") \
                .attr("Name", builtin_key(name))
            table_style_group_children(w, x, sub, tag, styles, seen)
            x.end()
        else:
            s = styles.get(c)
            if s is not None:
                ref = table_style_ref(w, tag, s)
                if ref not in seen:
                    seen.add(ref)
                    table_style_element(w, x, tag, s, styles)


def group_name(w, g):
    """The IDML name of a style group: the names of the groups on its
    path and its own."""
    from .idml_xml import style_path
    path = w.group_path.get(g.uid)
    if path is None:
        return style_path([g.name])
    return style_path(path)


def style_group_children(w, x, g, tag, sub_tag, written):
    """The paragraph or character styles and style groups of a style
    group, in its order."""
    for c in g.children:
        sub = w.doc.style_groups.get(c)
        if sub is not None:
            name = group_name(w, sub)
            x.start(sub_tag) \
                .attr("Self", f"{sub_tag}/$ID/{self_name(name)}") \
                .attr("Name", builtin_key(name))
            style_group_children(w, x, sub, tag, sub_tag, written)
            x.end()
        else:
            s = w.doc.styles.get(c)
            if s is not None:
                style_element(w, x, s, tag)
                written.add(c)


def inherited_list(w, s, attr_id):
    """Whether the list attribute ``attr_id`` (nested, line or GREP
    styles) of a paragraph style has items: the first style in its
    based-on chain that has the attribute decides. An empty list is
    stored as a 0 count."""
    cur = s
    for _ in range(32):
        if cur is None:
            break
        v = cur.attrs.get(attr_id)
        if v is not None:
            return v.kind == "style_list" and v.data[0] != 0
        cur = w.doc.styles.get(cur.based_on) \
            if cur.based_on is not None else None
    return False


def style_element(w, x, s, tag):
    """A paragraph or character style element."""
    doc = w.doc
    paragraph = s.paragraph
    name = w.grouped_name(s.uid, s.name, s.builtin)
    plain, props = text_attrs_of(w, s.attrs, not paragraph)
    if paragraph:
        # The schema allows KerningValue on character styles only.
        plain = [(k, v) for k, v in plain if k != "KerningValue"]
    is_root = s.builtin and s.based_on is None \
        and s.name.startswith("[No ")
    if is_root:
        # The root paragraph style has no `NextStyle` and no
        # `AllNestedStyles`, and an empty `TabList` only from DOM 8.
        props = [(n, t, v) for n, t, v in props if n != "AllNestedStyles"]
        root_lists(props, doc.version.major)
    x.start(tag) \
        .attr("Self", w.style_ref(s.uid, paragraph)) \
        .attr("Name", name)
    if paragraph and not is_root:
        nxt = s.next if s.next is not None else s.uid
        x.attr("NextStyle", w.style_ref(nxt, True))
    for k, v in plain:
        x.attr(k, v)
    x.attr("Imported", str(s.imported).lower())
    v = doc.version
    # From 11.2 every style has a unique ID: the stored GUID, else
    # `$ID/`.
    if w.saved_by((11, 2)):
        x.attr("StyleUniqueId",
               s.unique_id if s.unique_id is not None else "$ID/")
    if paragraph and (v.major >= 10 or (v.major == 8 and v.minor >= 1)):
        for attr_id, attr_name in [
                (0x1B75, "EmptyNestedStyles"),
                (0x1BBB, "EmptyLineStyles"),
                (0x1BBA, "EmptyGrepStyles"),
        ]:
            x.attr(attr_name, str(not inherited_list(w, s, attr_id)).lower())
    for name, value in export_flags(s.export, w.saved_by, True):
        x.attr(name, value)
    # Root styles also get the values every exported IDML has on them;
    # values read from the INDD take precedence.
    extra = []
    if is_root:
        written = [k for k, _v in plain]
        attrs, more, _children = root_values(w, tag, written, props)
        for k, v in attrs:
            if not x.has_attr(k):
                x.attr(k, v)
        extra = more
    else:
        short, extended = style_shortcut(s, doc.last_session_platform)
        if short is not None:
            x.attr("KeyboardShortcut", short)
        if v.major >= 15 and extended is not None:
            x.attr("ExtendedKeyboardShortcut", extended)
    if not is_root:
        n = values_element(tag, doc.version.major)
        if n is not None:
            # Other styles get the values every IDML has on them.
            x.attrs_missing(n.attrs)
            x.attrs_missing(values_when_written(tag, doc.version.major))
            p = n.child("Properties")
            if p is not None:
                extra = list(p.children)
    # The style's preview colour; the root styles have none.
    if not is_root and s.preview_color is not None:
        if s.preview_color is False:
            node = Node(tag="PreviewColor",
                        attrs=[("type", "enumeration")], text="Nothing")
        else:
            node = ui_color_property("PreviewColor", s.preview_color)
        if node is not None:
            found = next((e for e in extra if e.tag == "PreviewColor"),
                         None)
            if found is not None:
                extra[extra.index(found)] = node
            else:
                extra.append(node)
    if s.based_on is not None:
        base = doc.styles.get(s.based_on)
        if base is not None:
            # The root "[No ... style]" is written as a string.
            if base.builtin and base.name.startswith("[No "):
                props.insert(0, ("BasedOn", "string", ("text", style_name(base))))
            else:
                props.insert(0, ("BasedOn", "object",
                                 ("text", w.style_ref(base.uid, paragraph))))
    write_properties(x, props, extra)
    export_tag_maps(w, x, s)
    x.end()


def export_tag_maps(w, x, s):
    """The export tag maps of a paragraph or character style, with the
    style's split and CSS flags. Maps with attributes are left out:
    none occurs in the corpus."""
    if s.export is None:
        return
    flags = export_flags(s.export, w.saved_by, True)
    for m in s.export.maps:
        if m.attributes:
            continue
        x.start("StyleExportTagMap") \
            .attr("Self",
                  f"{uref(s.uid)}StyleExportTagMapn{m.export_type}") \
            .attr("ExportType", m.export_type) \
            .attr("ExportTag", m.tag) \
            .attr("ExportClass", m.class_name) \
            .attr("ExportAttributes", "")
        for name in ["SplitDocument", "EmitCss"]:
            f = next((v for n, v in flags if n == name), None)
            if f is not None:
                x.attr(name, f)
        x.end()


def export_flags(export, saved_by, text):
    """``SplitDocument``, ``EmitCss`` and ``IncludeClass`` of a
    paragraph or character style (``text``) or an object style: the
    first three u16 flags of chunk 0x28F0, or ``false``, ``true``,
    ``true`` without the chunk. IDML has ``SplitDocument`` (text
    styles only) and ``EmitCss`` when 10.1 saved the document,
    ``IncludeClass`` when 13.0 did."""
    flags = export.flags if export is not None else [0, 1, 1]
    since = [(10, 1), (10, 1), (13, 0)]
    out = []
    for name, since_v, v in zip(["SplitDocument", "EmitCss",
                                 "IncludeClass"], since, flags, strict=True):
        if not saved_by(since_v):
            continue
        if not text and name == "SplitDocument":
            continue
        if v == 0:
            out.append((name, "false"))
        elif v == 1:
            out.append((name, "true"))
    return out
