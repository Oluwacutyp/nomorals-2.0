"""``ObjectExportOption`` of page items and object styles.

A faithful port of ``src/idml/export.rs`` from storytold/indd-utils
(commit 136f973, dual MIT/Apache-2.0 — see attribution.py).
"""

from __future__ import annotations

from .idml_values import Node

#: The values every ``ObjectExportOption`` has, in IDML order, after
#: the alternative text and tagging values.
COMMON = [
    ("ImageConversionType", "JPEG"),
    ("ImageExportResolution", "Ppi300"),
    ("GIFOptionsPalette", "AdaptivePalette"),
    ("GIFOptionsInterlaced", "true"),
    ("JPEGOptionsQuality", "High"),
    ("JPEGOptionsFormat", "BaselineEncoding"),
    ("ImageAlignment", "AlignLeft"),
    ("ImageSpaceBefore", "0"),
    ("ImageSpaceAfter", "0"),
    ("UseImagePageBreak", "false"),
    ("ImagePageBreak", "PageBreakBefore"),
    ("CustomImageAlignment", "false"),
    ("SpaceUnit", "CssPixel"),
    ("CustomLayout", "false"),
    ("CustomLayoutType", "AlignmentAndSpacing"),
]


def _alt_source(code):
    return {0: "SourceCustom", 5: "SourceXMLStructure",
            8: "SourceDecorativeImage"}.get(code)


def _actual_source(code):
    return {0: "SourceCustom", 5: "SourceXMLStructure",
            6: "SourceXMPAltText"}.get(code)


def _tag_type(code):
    return {0: "TagFromStructure", 1: "TagArtifact"}.get(code)


def object_export_option(version, options, size_type, generated, owner):
    """The ``ObjectExportOption`` of a page item, a group or an object
    style, or ``None`` for versions whose IDML has none: page items
    before 8, object styles before 9. ``options`` are the item's values
    from chunk 0x1E206; without them the values of an item without the
    chunk apply. ``generated`` is the generated alternative text of an
    item other than a group. ``owner`` is ``"item"``, ``"group"`` or
    ``"style"``. Codes not identified leave their attribute out."""
    style = owner == "style"
    v = (version.major, version.minor)
    if v < (8, 0) or (style and v < (9, 0)):
        return None
    from .object import Name
    default_alt = Name(True, "")
    if options is None:
        class _Opts:
            pass
        o = _Opts()
        o.alt_source = 5
        o.actual_source = 5
        o.alt_text = default_alt
        o.actual_text = default_alt
        o.tag_type = 0
        o.alt_metadata = (default_alt, default_alt)
        o.actual_metadata = (default_alt, default_alt)
    else:
        o = options
    attrs = []

    def add(name, value):
        attrs.append((name, value))

    s = _alt_source(o.alt_source)
    if s is not None:
        add("AltTextSourceType", s)
    s = _actual_source(o.actual_source)
    if s is not None:
        add("ActualTextSourceType", s)
    add("CustomAltText", o.alt_text.idml())
    add("CustomActualText", o.actual_text.idml())
    s = _tag_type(o.tag_type)
    if s is not None:
        add("ApplyTagType", s)
    # 8.0 to 10.0 have CustomImageConversion; 8 and 9 also have
    # CustomImageSizeOption, which the schema the output is checked
    # against does not allow, so it is left out.
    old = v <= (10, 0)
    if old:
        add("CustomImageConversion", "false")
    for name, value in COMMON:
        add(name, value)
    if v == (10, 0):
        for name, value in [
                ("EpubType", "$ID/"),
                ("UseExistingImage", "false"),
                ("CustomHeightType", "DefaultHeight"),
                ("CustomHeight", "$ID/"),
                ("CustomWidthType", "DefaultWidth"),
                ("CustomWidth", "$ID/"),
        ]:
            add(name, value)
    elif v >= (10, 1):
        size = {2: "FixedSize", 3: "RelativeToTextFlow",
                4: "RelativeToTextSize"}.get(size_type, "DefaultSize")
        for name, value in [
                ("EpubType", "$ID/"),
                ("SizeType", size),
                ("CustomSize", "$ID/"),
                ("PreserveAppearanceFromLayout", "PreserveAppearanceDefault"),
        ]:
            add(name, value)
    if v >= (21, 0):
        add("EpubAriaRole", "$ID/")
    if v >= (21, 1):
        add("EpubAriaLabel", "$ID/")
        add("EpubAriaLabelSourceType", "AutomaticARIALabel")
    if v >= (21, 4):
        # Groups and styles have no generated alternative text.
        item = owner == "item"
        if item:
            add("AIGeneratedAltText", str(generated.ai).lower())
            add("AltTextGenerationError", str(generated.error).lower())
        if item and generated.crop is not None:
            add("AltTextCropSyncRect",
                ",".join(_eight_decimals(x) for x in generated.crop))
        else:
            add("AltTextCropSyncRect", "")
    if v >= (9, 2) and old:
        add("UseOriginalImage", "false")

    def metadata(tag, m):
        return Node(tag=tag, attrs=[
            ("NamespacePrefix", m[0].idml()),
            ("PropertyPath", m[1].idml()),
        ])

    empty = (default_alt, default_alt)
    if options is not None:
        alt, actual = o.alt_metadata, o.actual_metadata
    else:
        alt, actual = empty, empty
    return Node(tag="ObjectExportOption", attrs=attrs, children=[
        Node(tag="Properties", children=[
            metadata("AltMetadataProperty", alt),
            metadata("ActualMetadataProperty", actual),
        ])])


def _eight_decimals(v):
    """``v`` in its shortest decimal form, cut or padded to exactly 8
    decimals."""
    s = repr(float(v))
    if "e" in s or "E" in s:
        # Rust's Display for f64 never uses exponent form; expand.
        s = format(float(v), "f")
    int_part, dot, frac = s.partition(".")
    frac = (frac + "00000000")[:8]
    return f"{int_part}.{frac}"
