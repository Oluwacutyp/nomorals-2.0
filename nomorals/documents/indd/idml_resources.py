"""``Resources/Graphic.xml`` (colours, tints, gradients, inks, stroke
styles), ``Resources/Fonts.xml`` and ``Resources/Preferences.xml``.

A faithful port of ``src/idml/resources.rs`` from storytold/indd-utils
(commit 136f973, dual MIT/Apache-2.0 — see attribution.py).
"""

from __future__ import annotations

import struct

from .idml_attrs import (
    BUILTIN_STROKE_STYLES,
    FITTING_ATTRS,
    anchored_settings,
    contour_type,
    custom_stroke_ref,
    dash_written,
    end_cap,
    fitting_attrs,
    item_default_values,
    text_attrs,
)
from .idml_spread import object_style_ref
from .idml_styles import overprint_attrs, root_lists
from .idml_values import Node, keyed
from .idml_values import element as values_element
from .idml_values import preferences as values_preferences
from .idml_xml import (
    Xml,
    base64_lines,
    builtin_key,
    frame_grid_color,
    num,
    nums,
    prop_node,
    self_name,
    set_property,
    ui_color_property,
    uref,
)
from .idml_xml import round as round_half

#: Preference elements of ``Resources/Preferences.xml`` that
#: ``model::prefs`` supplies values for.
PREFERENCE_TAGS = [
    "ViewPreference",
    "MarginPreference",
    "TextFramePreference",
    "AnchoredObjectSetting",
    "GridPreference",
    "GuidePreference",
    "DocumentPreference",
    "TextPreference",
    "PasteboardPreference",
    "XMLPreference",
    "PrintPreference",
    "PrintBookletPrintPreference",
    "PrintBookletOption",
    "IndexHeaderSetting",
    "ChapterNumberPreference",
    "DictionaryPreference",
    "EPubExportPreference",
    "TransparencyPreference",
    "LayoutAdjustmentPreference",
    "StoryPreference",
    "TextWrapPreference",
    "CjkGridPreference",
    "MojikumiUiPreference",
    "XMLImportPreference",
    "HTMLExportPreference",
    "EPubFixedLayoutExportPreference",
]

#: Preference values that no INDD field holds and nearly every
#: reference IDML has: (element, attribute, value, first DOM version).
PREFERENCE_DEFAULTS = [
    ("PrintPreference", "TilingType", "Auto", (0, 0)),
    ("TextPreference", "HighlightKeeps", "false", (0, 0)),
    ("TextPreference", "HighlightSubstitutedFonts", "true", (0, 0)),
    ("ViewPreference", "GuideSnaptoZone", "4", (0, 0)),
    ("ViewPreference", "ShowNotes", "true", (0, 0)),
    ("XMLExportPreference", "ViewAfterExport", "false", (0, 0)),
    ("DocumentPreference", "SpreadHiddenVisibility", "false", (20, 0)),
]


def _pref_node(n):
    """A tree of preference values as a ``Node``."""
    return Node(tag=n.tag, attrs=list(n.attrs),
                children=[_pref_node(c) for c in n.children])


def group_swatches(w):
    """``SwatchColorGroupReference`` of each swatch in a colour group:
    the ``ColorGroupSwatch`` that names it."""
    out = {}
    for g in w.doc.color_groups:
        for n, s in enumerate(g.swatches):
            r = w.doc.swatches.get(s)
            if r is not None and r not in out:
                out[r] = f"{uref(g.uid)}ColorGroupSwatch{n:x}"
    return out


def graphic(w, refs):
    """``Resources/Graphic.xml``. ``refs`` are the references to
    unnamed colours and gradients in the other parts of the package:
    IDML writes an unnamed colour or gradient only when something
    refers to it or the page item defaults name it."""
    x = Xml()
    w.package_root(x, "Graphic")
    groups = group_swatches(w)

    def named(class_id):
        prefs = w.doc.prefs
        out = set()
        for e in prefs.item_default_entries:
            if e[0] == class_id:
                out.add(e[1])
                out.add(e[2])
        for e in prefs.item_default_swatches:
            if e[0] == class_id:
                out.add(e[1])
        return out

    # Class codes: GRADIENT 0x5503, COLOR 0x1F05, SMOOTH_SHADE 0x5533.
    gradients = [g for g in w.doc.gradients
                 if g.name or g.uid in named(0x5503)
                 or g.reference() in refs]
    # Colours that the stops of any gradient, written or not, and the
    # tints refer to.
    used = named(0x1F05)
    used.update(s.color for g in w.doc.gradients for s in g.stops)
    used.update(t.base for t in (t for t, _r, _n in w.doc.tints))
    colors = [c for c in w.doc.colors
              if c.name or c.uid in used or c.reference() in refs]

    def group_ref(reference):
        # IDML names the colour group swatch of every swatch from
        # DOM 12 on (`n` for none).
        if w.doc.version.major >= 12:
            x.attr("SwatchColorGroupReference",
                   groups.get(reference, "n"))

    for c in colors:
        name = c.idml_name()
        attrs = [("Self", c.reference())]
        model = c.model_name()
        if model is not None:
            attrs.append(("Model", model))
        attrs.extend([
            ("Space", c.space_name()),
            ("ColorValue", nums(c.idml_values())),
            ("ColorOverride", c.override_name()),
        ])
        # Values that follow the model.
        keyed_attrs = []
        if model is not None:
            n = keyed("Color", model)
            if n is not None:
                keyed_attrs = [(k, v) for k, v in n.attrs if k != "Model"]
        attrs.extend(keyed_attrs)
        attrs.extend([
            ("Name", name),
            ("ColorEditable", str(c.editable).lower()),
            ("ColorRemovable", str(c.removable).lower()),
            ("Visible", str(c.visible).lower()),
        ])
        alt = c.idml_alternate()
        if alt is not None:
            space, values = alt
            attrs.append(("AlternateSpace", space))
            attrs.append(("AlternateColorValue", nums(values)))
        if c.creator is not None:
            attrs.append(("SwatchCreatorID", str(c.creator)))
        if w.doc.version.major >= 16:
            hsb = c.space == ("hsb",)
            attrs.append(("ConvertToHsb", str(hsb).lower()))
        x.start("Color")
        for k, v in attrs:
            x.attr(k, v)
        group_ref(c.reference())
        x.end()
    # The schema puts inks after the colours.
    for i in w.doc.inks:
        attrs = [
            ("Self", f"Ink/{self_name(i.name.idml())}"),
            ("Name", i.name.idml()),
            ("Angle", num(i.angle)),
        ]
        if i.convert_to_process is not None:
            attrs.append(("ConvertToProcess",
                          str(i.convert_to_process).lower()))
        attrs.append(("Frequency", num(i.frequency)))
        if i.neutral_density is not None:
            attrs.append(("NeutralDensity", num(i.neutral_density)))
        attrs.append(("TrapOrder", str(i.trap_order)))
        x.start("Ink")
        for k, v in attrs:
            x.attr(k, v)
        # The values every IDML has on an ink.
        x.attrs_missing(w.observed("Ink"))
        x.end()
    # Mixed ink groups, then mixed inks sorted by name in code point
    # order.
    mixed = sorted(w.doc.mixed_inks,
                   key=lambda m: (not m.group, m.name))
    for m in mixed:
        source = next((n for n in w.doc.mixed_inks if n.uid == m.base),
                      m) if m.base is not None else m
        inks = [i for u in source.inks
                for i in [next((i for i in w.doc.inks if i.uid == u), None)]
                if i is not None]
        spots = [c for u in source.spots
                 for c in [next((c for c in w.doc.colors if c.uid == u), None)]
                 if c is not None]
        if len(inks) != len(source.inks) or len(spots) != len(source.spots):
            continue
        reference = m.reference()
        x.start("MixedInkGroup" if m.group else "MixedInk") \
            .attr("Self", reference) \
            .attr("Model", "Mixedinkmodel")
        if not m.group:
            x.attr("Space", "MixedInk")
        x.attr("InkList", " ".join(
            f"Ink/{self_name(i.name.idml()).replace(' ', '%20')}"
            for i in inks))
        if not m.group:
            x.attr("InkPercentages", " ".join(
                str(p * 100.0) for p in m.percentages))
            base = w.doc.swatches.get(m.base, "n") \
                if m.base is not None else "n"
            x.attr("BaseColor", base)
        x.attr("InkNameList", " ".join(
            i.name.idml().replace(" ", "%20") for i in inks)) \
            .attr("MixedInkSpotColorNameList", " ".join(
                c.idml_name().replace(" ", "%20") for c in spots)) \
            .attr("MixedInkSpotColorList", " ".join(
                c.reference().replace(" ", "%20") for c in spots)) \
            .attr("Name", m.name) \
            .attr("ColorEditable", str(m.editable).lower()) \
            .attr("ColorRemovable", str(m.removable).lower()) \
            .attr("Visible", str(m.visible).lower())
        if m.creator is not None:
            x.attr("SwatchCreatorID", str(m.creator))
        group_ref(reference)
        x.end()
    # The pasted smooth shades, in UID order: the document's constant
    # shade, and each axial shade that something refers to or the page
    # item defaults name. Each has the values every IDML has on one of
    # its kind; its contents are the stored fields in big-endian order.
    named_shades = named(0x5533)
    # UID, contents type, matrix, contents, name.
    shades = []
    if w.doc.constant_shade is not None:
        shade = w.doc.constant_shade
        data = struct.pack(">I", shade.type_code)
        for v in shade.values:
            data += struct.pack(">d", v)
        shades.append((shade.uid, "ConstantShade", None, data,
                       (shade.builtin, shade.name)
                       if shade.name is not None else None))
    for a in w.doc.axial_shades:
        reference = f"PastedSmoothShade/{uref(a.uid)}"
        if reference in refs or a.uid in named_shades:
            shades.append((a.uid, "AxialShade", a.matrix, a.data,
                           (a.builtin, a.name)
                           if a.name is not None else None))
    shades.sort(key=lambda s: s[0])
    for uid, kind, matrix, data, name in shades:
        node = keyed("PastedSmoothShade", kind)
        if node is None:
            continue
        reference = f"PastedSmoothShade/{uref(uid)}"
        x.start("PastedSmoothShade") \
            .attr("Self", reference) \
            .attr("ContentsType", kind)
        if matrix is not None:
            x.attr("ContentsMatrix", nums(matrix))
        if name is not None:
            builtin, n = name
            x.attr("Name", builtin_key(n) if builtin else n)
        x.attrs_missing(node.attrs)
        group_ref(reference)
        x.start("Properties") \
            .start("Contents") \
            .cdata(base64_lines(data), 262144) \
            .end() \
            .end()
        x.end()
    for t, reference, name in w.doc.tints:
        base = w.doc.swatches.get(t.base, "")
        x.start("Tint") \
            .attr("Self", reference) \
            .attr("TintValue", num(t.value)) \
            .attr("BaseColor", base) \
            .attr("Name", name) \
            .attr("ColorOverride", t.override_name())
        # The alternate colour of the base colour.
        alt = next((c.idml_alternate() for c in w.doc.colors
                    if c.uid == t.base), None)
        if alt is not None:
            space, values = alt
            x.attr("AlternateSpace", space) \
                .attr("AlternateColorValue", nums(values))
        group_ref(reference)
        x.attrs_missing(w.observed("Tint"))
        x.end()
    x.start("Swatch") \
        .attr("Self", "Swatch/None") \
        .attr("Name", "None") \
        .attr("ColorEditable", "false") \
        .attr("ColorRemovable", "false") \
        .attr("Visible", "true")
    group_ref("Swatch/None")
    x.attrs_missing(w.observed("Swatch"))
    x.end()
    # The schema requires gradients after the swatches.
    for g in gradients:
        x.start("Gradient") \
            .attr("Self", g.reference()) \
            .attr("Type", "Radial" if g.kind == 2 else "Linear") \
            .attr("Name", g.idml_name()) \
            .attr("ColorEditable", str(g.editable).lower()) \
            .attr("ColorRemovable", str(g.removable).lower()) \
            .attr("Visible", str(g.visible).lower())
        group_ref(g.reference())
        x.attrs_missing(w.observed("Gradient"))
        for i, stop in enumerate(g.stops):
            color = w.doc.swatches.get(stop.color, "Color/Black")
            x.start("GradientStop") \
                .attr("Self", f"{uref(g.uid)}GradientStop{i}") \
                .attr("StopColor", color) \
                .attr("Location", num(round_half(stop.location * 100.0)))
            m = g.idml_midpoint(i)
            if m is not None:
                x.attr("Midpoint", num(m))
            x.end()
        x.end()
    for name in BUILTIN_STROKE_STYLES:
        x.empty("StrokeStyle", [
            ("Self", f"StrokeStyle/$ID/{name}"),
            ("Name", builtin_key(name)),
        ])
    # Custom stroke styles, in UID order.
    for st in w.doc.stroke_styles:
        if st.dashed:
            x.start("DashedStrokeStyle") \
                .attr("Self", custom_stroke_ref(st)) \
                .attr("DashArray", nums(st.values))
            if st.corner == 3:
                x.attr("StrokeCornerAdjustment", "DashesAndGaps")
            cap = end_cap(st.cap)
            if cap is not None:
                x.attr("EndCap", cap)
        else:
            stripes = [v * 100.0 for v in st.values]
            x.start("StripedStrokeStyle") \
                .attr("Self", custom_stroke_ref(st)) \
                .attr("StripeArray", nums(stripes))
        x.attr("Name", st.name).end()
    x.end()
    return x.finish()


def fonts(w):
    """Font families and their fonts."""
    x = Xml()
    w.package_root(x, "Fonts")
    # InDesign 7.0 up to build 522 writes every PostScript name as a
    # built-in key; builds 535 and later do not.
    postscript_key = w.doc.last_session_version == (7, 0) \
        and w.doc.last_session_build is not None \
        and w.doc.last_session_build <= 522
    # Font families in UID order.
    for f in sorted(w.doc.fonts.values(), key=lambda f: f.uid):
        ident = f"di{f.uid:x}"
        family, base = w.family_names(f)
        x.start("FontFamily") \
            .attr("Self", ident) \
            .attr("Name", family)
        # From DOM 21 a font's `Name` drops the spaces that end its
        # family's name; its `Self` keeps them.
        named = family.rstrip() if w.doc.version.major >= 21 else family
        for font in f.fonts:
            # A font with an empty style name is named `Regular`, and
            # its `Self` has the family name alone.
            if not font.style:
                name, key = f"{named} Regular", base
            else:
                name, key = f"{named} {font.style}", f"{base} {font.style}"
            style = builtin_key(font.style) if font.style_builtin \
                else font.style
            x.start("Font") \
                .attr("Self", f"{ident}Fontn{key}") \
                .attr("FontFamily", base) \
                .attr("Name", name) \
                .attr("PostScriptName",
                      builtin_key(font.postscript_name)
                      if postscript_key and font.postscript_name
                      else font.postscript_name) \
                .attr("FontStyleName", style)
            t = font.type_name()
            if t is not None:
                x.attr("FontType", t)
            # The schema's `xsd:int`; one file stores 0xFFFFFFFF.
            try:
                w32 = f.writing_script
                if -(2 ** 31) <= w32 < 2 ** 31:
                    x.attr("WritingScript", str(w32))
            except (TypeError, ValueError):
                pass
            x.attr("FullName", font.full_name) \
                .attr("FullNameNative", font.full_name_native) \
                .attr("FontStyleNameNative", font.style_native) \
                .attr("PlatformName",
                      font.platform_name
                      if font.platform_name is not None else "$ID/") \
                .attr("Version", font.version)
            # IDML from InDesign before 9.2 has no TypekitID.
            v = w.doc.version
            if (v.major, v.minor) >= (9, 2):
                x.attr("TypekitID", font.typekit_id.idml())
            x.end()
        x.end()
    # Composite fonts in UID order.
    for cf in sorted(w.doc.composite_fonts, key=lambda c: c.uid):
        name = cf.name.idml()
        x.start("CompositeFont") \
            .attr("Self", f"CompositeFont/{self_name(name)}") \
            .attr("Name", name)
        for k, e in enumerate(cf.entries):
            size, shift, horizontal, vertical = e.numbers
            x.start("CompositeFontEntry") \
                .attr("Self", uref(e.uid)) \
                .attr("Name", e.name.idml()) \
                .attr("FontStyle", e.font_style.idml()) \
                .attr("RelativeSize", num(size)) \
                .attr("HorizontalScale", num(horizontal)) \
                .attr("VerticalScale", num(vertical))
            # The first entry is the base font, without characters.
            if k > 0:
                x.attr("CustomCharacters", e.characters())
            if e.locked == 0:
                x.attr("Locked", "false")
            elif e.locked == 1:
                x.attr("Locked", "true")
            if e.scale == [1, 1, 1, 1]:
                x.attr("ScaleOption", "true")
            elif e.scale == [0, 0, 0, 0]:
                x.attr("ScaleOption", "false")
            x.attr("BaselineShift", num(shift))
            f = w.doc.fonts.get(e.font_family)
            if f is not None:
                from .idml_xml import write_properties
                write_properties(x, [("AppliedFont", "string",
                                      ("text", w.family_names(f)[0]))])
            x.end()
        x.end()
    x.end()
    return x.finish()


def footnote_option(w):
    """``FootnoteOption`` from chunk 0x2820. The values that never vary
    come from the generated value files."""
    from .idml_attrs import STROKE_TYPES

    major = w.doc.version.major
    n = Node(tag="FootnoteOption")
    f = w.doc.prefs.footnotes
    if f is None:
        # Without the chunk every IDML has these values.
        if major >= 12:
            n.attrs.append(("EnableStraddling", "true"))
        n.attrs.append(("FootnoteTextStyle", w.style_ref(None, True)))
        for k, v in [
                ("StartAt", "1"),
                ("Prefix", ""),
                ("Suffix", ""),
                ("FootnoteMarkerStyle",
                 "CharacterStyle/$ID/[No character style]"),
                ("SeparatorText", "\t"),
                ("SpaceBetween", "0"),
                ("Spacer", "0"),
                ("FootnoteMinimumFirstBaselineOffset", "0"),
                ("EosPlacement", "false"),
                ("NoSplitting", "false"),
        ]:
            n.attrs.append((k, v))
        for prefix in ["Rule", "ContinuingRule"]:
            for k, v in [
                    ("On", "true"),
                    ("LineWeight", "1"),
                    ("Tint", "100"),
                    ("GapTint", "100"),
                    ("LeftIndent", "0"),
                    ("Width", "72" if prefix == "Rule" else "288"),
                    ("Offset", "0"),
            ]:
                n.attrs.append((f"{prefix}{k}", v))

        def text_prop(name, ty, text):
            return Node(tag=name, attrs=[("type", ty)], text=text)

        props = [text_prop(k, "enumeration", v) for k, v in [
            ("FootnoteNumberingStyle", "Arabic"),
            ("RestartNumbering", "DontRestart"),
            ("ShowPrefixSuffix", "NoPrefixSuffix"),
            ("MarkerPositioning", "SuperscriptMarker"),
        ]]
        for prefix in ["Rule", "ContinuingRule"]:
            for k, v in [
                    ("Type", "StrokeStyle/$ID/Solid"),
                    ("Color", "Color/Black"),
                    ("GapColor", "Swatch/None"),
            ]:
                props.append(text_prop(f"{prefix}{k}", "object", v))
        n.children.append(Node(tag="Properties", children=props))
        return n
    if major >= 12 and f.tail is not None and f.tail.straddling is not None:
        n.attrs.append(("EnableStraddling", str(f.tail.straddling).lower()))
    n.attrs.append(("StartAt", str(f.start_at)))
    n.attrs.append(("Prefix", f.prefix))
    n.attrs.append(("Suffix", f.suffix))
    n.attrs.append(("FootnoteTextStyle", w.style_ref(f.text_style, True)))
    n.attrs.append(("FootnoteMarkerStyle", w.style_ref(f.marker_style, False)))
    n.attrs.append(("SeparatorText", f.separator))
    n.attrs.append(("SpaceBetween", num(f.space_between)))
    n.attrs.append(("Spacer", num(f.spacer)))
    if f.tail is not None:
        n.attrs.append(("FootnoteMinimumFirstBaselineOffset",
                        num(f.tail.minimum_first_baseline)))
    if f.eos_placement == 0:
        n.attrs.append(("EosPlacement", "false"))
    elif f.eos_placement == 1:
        n.attrs.append(("EosPlacement", "true"))
    props = []

    def prop(name, ty, text):
        props.append(Node(tag=name, attrs=[("type", ty)], text=text))

    numbering = {0xCA07: "Arabic", 0xCA0A: "Symbols",
                 0xCA58: "Asterisks"}.get(f.numbering)
    if numbering is not None:
        prop("FootnoteNumberingStyle", "enumeration", numbering)
    restart = {0: "DontRestart", 1: "PageRestart"}.get(f.restart)
    if restart is not None:
        prop("RestartNumbering", "enumeration", restart)
    prefix_suffix = {0: "NoPrefixSuffix", 1: "PrefixSuffixReference",
                     3: "PrefixSuffixBoth"}.get(f.prefix_suffix)
    if prefix_suffix is not None:
        prop("ShowPrefixSuffix", "enumeration", prefix_suffix)
    marker = {(1, 0): "SuperscriptMarker", (0, 0): "NormalMarker",
              (0, 1): "RubyMarker"}.get(f.marker)
    if marker is not None:
        prop("MarkerPositioning", "enumeration", marker)
    if f.tail is not None:
        n.attrs.append(("NoSplitting", str(f.tail.no_splitting).lower()))
        for r, name in zip(f.tail.rules, ["Rule", "ContinuingRule"],
                           strict=True):
            n.attrs.append((f"{name}On", str(r.on).lower()))
            n.attrs.append((f"{name}LineWeight", num(r.weight)))
            n.attrs.append((f"{name}Tint", num(r.tint)))
            n.attrs.append((f"{name}GapTint", num(r.gap_tint)))
            n.attrs.append((f"{name}LeftIndent", num(r.left_indent)))
            n.attrs.append((f"{name}Width", num(r.width)))
            n.attrs.append((f"{name}Offset", num(r.offset)))
            stroke = next((s for k, s in STROKE_TYPES if k == r.stroke), None)
            if stroke is not None:
                prop(f"{name}Type", "object", f"StrokeStyle/$ID/{stroke}")
            for uid, suffix in [(r.color, "Color"),
                                (r.gap_color, "GapColor")]:
                sw = w.doc.swatches.get(uid)
                if sw is not None:
                    prop(f"{name}{suffix}", "object", sw)
    if props:
        n.children.append(Node(tag="Properties", children=props))
    return n


def index_options(w, o):
    """The attributes of ``IndexOptions``. A style is named by its
    name alone, among the styles outside style groups; without such a
    style, IDML names the root style."""
    def style(name, paragraph):
        found = next((s for s in w.doc.styles.values()
                      if s.paragraph == paragraph and s.name == name
                      and not w.group_path.get(s.uid)), None)
        if found is not None:
            return w.style_ref(found.uid, paragraph)
        if paragraph:
            return "ParagraphStyle/$ID/[No paragraph style]"
        return "CharacterStyle/$ID/[No character style]"

    out = [
        ("Title", o.title),
        ("TitleStyle", style(o.title_style, True)),
        ("ReplaceExistingIndex", str(o.replace_existing).lower()),
        ("IncludeBookDocuments", str(o.include_book_documents).lower()),
        ("IncludeSectionHeadings",
         str(o.include_section_headings).lower()),
    ]
    for name, text in zip([
            "FollowingTopicSeparator",
            "BetweenPageNumbersSeparator",
            "BetweenEntriesSeparator",
            "BeforeCrossReferenceSeparator",
            "PageRangeSeparator",
            "EntryEndSeparator",
    ], o.separators, strict=True):
        # IDML writes the en dash of the page range as `^=`.
        if name == "PageRangeSeparator":
            text = text.replace("\u2013", "^=")
        out.append((name, text))
    for i, (name, s) in enumerate(zip([
            "Level1Style",
            "Level2Style",
            "Level3Style",
            "Level4Style",
            "SectionHeadingStyle",
            "PageNumberStyle",
            "CrossReferenceStyle",
            "CrossReferenceTopicStyle",
    ], o.styles, strict=True)):
        out.append((name, style(s, i < 5)))
    return out


def index_title(w):
    """The index title, where one is known, and the separator between
    entries of a document without index options. Both follow the
    edition that made the document."""
    # The title follows the edition that saved the document, where its
    # black name shows it.
    saved = None
    edition = w.saving_edition()
    if edition is not None:
        saved = {
            "english": "Index", "german": "Index", "french": "Index",
            "dutch": "Index", "korean": "색인",
            "japanese": "索引", "chinese": "索引",
            "italian": "Indice", "czech": "Rejstřík",
            "portuguese": "Índice", "russian": "Указатель",
        }.get(edition)
    edition = w.edition()
    if edition is None:
        e = w.saving_edition()
        if e is None:
            return None
        separator = "、" if e == "japanese" else "; "
        return saved, separator
    title, separator = _assigned_index_title(w, edition)
    return (saved if saved is not None else title), separator


def _assigned_index_title(w, edition):
    """The index title and separator by the edition of the assignment
    name."""
    japanese_session = w.japanese_session() is True
    if edition == "japanese":
        return "索引", "、"
    if edition == "chinese":
        return "索引", "; "
    if edition == "korean":
        return "색인", "; "
    if edition == "english" and japanese_session:
        return None, "; "
    if edition in ("english", "french", "german", "dutch"):
        return "Index", "; "
    if edition == "italian":
        return "Indice", "; "
    return None, "; "


def preferences(w):
    """``Resources/Preferences.xml``."""
    x = Xml()
    w.package_root(x, "Preferences")
    # Values every exported IDML has; values read from the INDD take
    # precedence.
    nodes = values_preferences(w.doc.version.major)
    # DOM 7 IDML has no EPUB and HTML export preferences.
    if w.doc.version.major < 8:
        nodes = [n for n in nodes
                 if n.tag not in ("EPubExportPreference",
                                  "HTMLExportPreference")]
    # Values every IDML of the version has in the larger corpus, where
    # the file above has none.
    for n in nodes:
        e = values_element(f"Preferences/{n.tag}", w.doc.version.major)
        if e is not None:
            n.merge(e)
    # Object styles named by the page item defaults must be in the
    # package.
    styles = {r for u in w.doc.object_styles
              for r in [object_style_ref(w, u)] if r is not None}
    for n in nodes:
        if n.tag == "PageItemDefault":
            n.attrs = [(k, v) for k, v in n.attrs
                       if not k.endswith("ObjectStyle") or v in styles]
    # Observed values the INDD contradicts but the converter cannot
    # replace: (element, attribute).
    drop = []

    def set_node(tag, ours):
        i = next((i for i, n in enumerate(nodes) if n.tag == tag), None)
        if i is None:
            nodes.append(Node(tag=tag))
            i = len(nodes) - 1
        node = nodes[i]
        node.attrs = [(k, v) for k, v in node.attrs
                      if k not in {o for o, _v in ours}]
        rest = node.attrs
        node.attrs = [(k, v) for k, v in ours]
        node.attrs.extend(rest)

    p = w.doc.preferences
    if p is not None:
        top, bottom, inside, outside = p.bleed
        ours = [
            ("PageHeight", num(p.page_height)),
            ("PageWidth", num(p.page_width)),
            ("FacingPages", str(p.facing_pages).lower()),
            ("DocumentBleedTopOffset", num(top)),
            ("DocumentBleedBottomOffset", num(bottom)),
            ("DocumentBleedInsideOrLeftOffset", num(inside)),
            ("DocumentBleedOutsideOrRightOffset", num(outside)),
        ]
        intent = ["PrintIntent", "WebIntent",
                  "MobileIntent"][p.intent] if p.intent in (0, 1, 2) \
            else None
        if intent is not None:
            ours.append(("Intent", intent))
        # Every sample's IDML has `LeftToRight`; a code the samples do
        # not show leaves the attribute out.
        if p.page_binding == 0:
            ours.append(("PageBinding", "LeftToRight"))
        elif p.page_binding == 1:
            ours.append(("PageBinding", "RightToLeft"))
        else:
            drop.append(("DocumentPreference", "PageBinding"))
        set_node("DocumentPreference", ours)
    # Guide locations are measured from the ruler origin.
    from .model import RulerOrigin
    set_node("ViewPreference",
             [("RulerOrigin", RulerOrigin.idml(w.doc.prefs.ruler_origin))])
    # Values read from the preferences object (`model::prefs`).
    prefs = w.doc.prefs
    ours_nodes = []

    def ours_of(tag):
        i = next((i for i, n in enumerate(ours_nodes) if n.tag == tag),
                 None)
        if i is None:
            ours_nodes.append(Node(tag=tag))
            i = len(ours_nodes) - 1
        return i

    for v in prefs.values:
        if v.element in PREFERENCE_TAGS:
            i = ours_of(v.element)
            ours_nodes[i].attrs.append((v.name, v.value))
    # The column rule (DOM 15 on) and footnote values (13.1 on) of the
    # default text frame settings, as on frames.
    version = (w.doc.version.major, w.doc.version.minor)
    if version[0] >= 15:
        i = ours_of("TextFramePreference")
        if prefs.frame_column_rule is not None:
            width, color_uid = prefs.frame_column_rule
            color = "n" if color_uid == 0 \
                else w.doc.swatches.get(color_uid)
        else:
            width, color = 1.0, "Color/Black"
        ours_nodes[i].attrs.append(("ColumnRuleStrokeWidth", num(width)))
        if color is not None:
            ours_nodes[i].attrs.append(("ColumnRuleStrokeColor", color))
    if w.saved_by((13, 1)):
        i = ours_of("TextFramePreference")
        span, minimum, between = prefs.frame_footnotes \
            if prefs.frame_footnotes is not None else (False, 12.0, 6.0)
        ours_nodes[i].attrs.extend([
            ("FootnotesEnableOverrides", "false"),
            ("FootnotesSpanAcrossColumns", str(span).lower()),
            ("FootnotesMinimumSpacing", num(minimum)),
            ("FootnotesSpaceBetween", num(between)),
        ])
    if prefs.anchor is not None:
        i = ours_of("AnchoredObjectSetting")
        ours_nodes[i].attrs.extend(anchored_settings(prefs.anchor))
    if prefs.item_defaults is not None:
        a = prefs.item_defaults
        i = ours_of("PageItemDefault")
        values = item_default_values(w, a)
        stroke = next((v for k, v in values if k == "StrokeType"), None)
        values = [(k, v) for k, v in values if dash_written(k, stroke)]
        values.extend(overprint_attrs(w, a, False))
        ours_nodes[i].attrs.extend(values)
        all_ids = [attr_id for attr_id, _n, _k in FITTING_ATTRS]
        fitting = fitting_attrs(w, a, all_ids)
        # Every IDML has all seven; a value the list lacks is code or
        # number 0.
        for (attr_id, name, _k), zero in zip(
                FITTING_ATTRS,
                ["false", "0", "0", "0", "0", "None", "TopLeftAnchor"],
                strict=True):
            if a.get(attr_id) is None:
                fitting.append((name, zero))
        i = ours_of("FrameFittingOption")
        ours_nodes[i].attrs.extend(fitting)
    for tag, name, data in prefs.print_records:
        text = base64_lines(data)
        # IDML ends a last line of full length with a line feed.
        if text.rsplit("\n", 1)[-1] and len(text.rsplit("\n", 1)[-1]) == 76:
            text += "\n"
        i = ours_of(tag)
        ours_nodes[i].attrs.append((name, builtin_key(text)))
    for tag, name, value in prefs.props:
        kind = value[0]
        n = Node(tag=name)
        if kind == "text":
            _k, ty, text = value
            n.attrs.append(("type", ty))
            n.text = text
        elif kind == "attrs":
            _k, attrs = value
            n.attrs = list(attrs)
        elif kind == "nodes":
            _k, children = value
            n.children = [_pref_node(c) for c in children]
        i = ours_of(tag)
        props = next((c for c in ours_nodes[i].children
                      if c.tag == "Properties"), None)
        if props is not None:
            props.children.append(n)
        else:
            ours_nodes[i].children.insert(
                0, Node(tag="Properties", children=[n]))
    for tag, name, rgb in prefs.colors:
        if tag not in PREFERENCE_TAGS:
            continue
        prop = ui_color_property(name, rgb)
        if prop is None:
            continue
        i = ours_of(tag)
        props = next((c for c in ours_nodes[i].children
                      if c.tag == "Properties"), None)
        if props is not None:
            props.children.append(prop)
        else:
            ours_nodes[i].children.insert(
                0, Node(tag="Properties", children=[prop]))
    if prefs.text_defaults is not None:
        plain, props = text_attrs(w, prefs.text_defaults)
        # The schema allows KerningValue on character styles only.
        plain = [(k, v) for k, v in plain if k != "KerningValue"]
        root_lists(props, w.doc.version.major)
        i = ours_of("TextDefault")
        ours_nodes[i].attrs.extend(plain)
        if props:
            ours_nodes[i].children.append(Node(
                tag="Properties",
                children=[prop_node(p) for p in props]))
    # Default styles.
    for uid, paragraph, name in [
            (prefs.default_styles[0], True, "AppliedParagraphStyle"),
            (prefs.default_styles[1], False, "AppliedCharacterStyle"),
    ]:
        value = None
        if uid is not None:
            s = w.doc.styles.get(uid)
            if s is not None and s.paragraph == paragraph:
                value = w.style_ref(uid, paragraph)
            elif uid == 0 and paragraph:
                value = "ParagraphStyle/$ID/[No paragraph style]"
            elif uid == 0:
                value = "CharacterStyle/$ID/[No character style]"
        if value is not None:
            i = ours_of("TextDefault")
            ours_nodes[i].attrs.append((name, value))
    if prefs.default_object_styles is not None:
        for u, name in zip(prefs.default_object_styles, [
                "AppliedGraphicObjectStyle",
                "AppliedTextObjectStyle",
                "AppliedGridObjectStyle",
        ], strict=True):
            r = object_style_ref(w, u)
            if r is not None:
                i = ours_of("PageItemDefault")
                ours_nodes[i].attrs.append((name, r))
    for tag, g, count in prefs.grids:
        i = ours_of(tag)
        n = ours_nodes[i]
        n.attrs.append(("FontStyle", g.font_style))
        size, character_aki, line_aki, h_scale, v_scale = g.numbers
        for name, v in [
                ("PointSize", size),
                ("CharacterAki", character_aki),
                ("LineAki", line_aki),
                ("HorizontalScale", h_scale * 100.0),
                ("VerticalScale", v_scale * 100.0),
        ]:
            n.attrs.append((name, num(v)))
        if count is not None:
            n.attrs.append(("CharacterCountSize", num(count)))
        f = w.doc.fonts.get(g.font)
        if f is not None:
            prop = prop_node(("AppliedFont", "string",
                              ("text", w.family_names(f)[0])))
            props = next((c for c in n.children if c.tag == "Properties"),
                         None)
            if props is not None:
                props.children.append(prop)
            else:
                n.children.insert(0, Node(tag="Properties",
                                          children=[prop]))
    if prefs.index_options is not None:
        i = ours_of("IndexOptions")
        ours_nodes[i].attrs.extend(index_options(w, prefs.index_options))
    else:
        # Without the chunk every IDML has these values; the title and
        # the separator follow the language of the last session. Other
        # languages have different titles.
        i = ours_of("IndexOptions")
        paragraph = "ParagraphStyle/$ID/[No paragraph style]"
        character = "CharacterStyle/$ID/[No character style]"
        for k, v in [
                ("TitleStyle", paragraph),
                ("ReplaceExistingIndex", "true"),
                ("IncludeBookDocuments", "false"),
                ("IncludeHiddenEntries", "false"),
                ("IndexFormat", "NestedFormat"),
                ("IncludeSectionHeadings", "true"),
                ("IncludeEmptyIndexSections", "false"),
                ("Level1Style", paragraph),
                ("Level2Style", paragraph),
                ("Level3Style", paragraph),
                ("Level4Style", paragraph),
                ("SectionHeadingStyle", paragraph),
                ("PageNumberStyle", character),
                ("CrossReferenceStyle", character),
                ("CrossReferenceTopicStyle", character),
                ("FollowingTopicSeparator", "  "),
                ("PageRangeSeparator", "^="),
                ("BetweenPageNumbersSeparator", ", "),
                ("BeforeCrossReferenceSeparator", ". "),
                ("EntryEndSeparator", ""),
        ]:
            ours_nodes[i].attrs.append((k, v))
        title_sep = index_title(w)
        if title_sep is not None:
            title, separator = title_sep
            if title is not None:
                ours_nodes[i].attrs.append(("Title", title))
            ours_nodes[i].attrs.append(("BetweenEntriesSeparator",
                                        separator))
    # Without chunk 0x7006 the snap zone follows the ruler units and
    # the edition that made the document.
    if not any(v.element == "LayoutAdjustmentPreference"
               and v.name == "SnapZone" for v in prefs.values):
        edition = w.edition()
        if edition is not None:
            metric = any(v.element == "ViewPreference"
                         and v.name == "HorizontalMeasurementUnits"
                         and v.value in ("Millimeters", "Centimeters")
                         for v in prefs.values)
            zone = "0.70866141732283" \
                if metric or edition != "english" else "2"
            i = ours_of("LayoutAdjustmentPreference")
            ours_nodes[i].attrs.append(("SnapZone", zone))
    # The contour of new graphics.
    if prefs.contour is not None:
        t = contour_type(prefs.contour.kind)
        if t is not None:
            i = ours_of("TextWrapPreference")
            attrs = [("ContourType", t)]
            if w.dom_version() >= (21, 4):
                from .idml_attrs import CONTOUR_PARAMS
                attrs.extend(CONTOUR_PARAMS)
            ours_nodes[i].set(["ContourOption"], attrs)
    ours_nodes.append(footnote_option(w))
    if prefs.baseline_frame_grid_color is not None:
        c = frame_grid_color(prefs.baseline_frame_grid_color)
        if c is not None:
            i = ours_of("BaselineFrameGridOption")
            set_property(ours_nodes[i], c)
    for n in ours_nodes:
        j = next((j for j, m in enumerate(nodes) if m.tag == n.tag), None)
        if j is not None:
            n.merge(nodes[j])
            nodes[j] = n
        else:
            nodes.append(n)
    for tag, attr in drop:
        for n in nodes:
            if n.tag == tag:
                n.attrs = [(k, v) for k, v in n.attrs if k != attr]
    for tag, name, value, from_v in PREFERENCE_DEFAULTS:
        if w.dom_version() < from_v:
            continue
        n = next((n for n in nodes if n.tag == tag), None)
        if n is not None and n.attr(name) is None:
            n.attrs.append((name, value))
    for n in nodes:
        n.write(x)
    x.end()
    return x.finish()
