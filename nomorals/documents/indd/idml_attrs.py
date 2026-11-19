"""IDML attribute tables and decoders.

A faithful port of ``src/idml/attrs.rs`` (tables) and the value-kind
conversion of ``src/idml/kind.rs`` from storytold/indd-utils (commit
136f973, dual MIT/Apache-2.0 — see attribution.py). The tables were
translated mechanically from the Rust literals (see
~/workspace/inddutils-verify/translate_attrs.py) and spot-checked
against the source; value kinds are tuples: ("number",),
("range", lo, hi), ("enum", ((code, name), ...)), and so on.

A ``Property`` is (name, type, value) where value is one of
("text", s), ("list", [[(field, type, text), ...], ...]),
("lists", [[(type, text), ...], ...]) or ("attrs", [(k, v), ...]).
"""

from __future__ import annotations

from .idml_xml import num, nums, self_name
from .model import FormKind

#: Value of the kerning attribute (0x1B13) for no manual kerning; IDML
#: writes it ``1e+11``. Every root paragraph style stores it.
KERNING_NONE = 1e8

#: The stroke type code of a custom stroke style, stored with its UID.
CUSTOM_STROKE_TYPE = 0x5A42

#: Stroke style code of a cell edge that has no stroke type (IDML ``n``).
CELL_NO_STROKE_TYPE = 0x1040C

#: Cell style paragraph style attribute.
CELL_STYLE_PARAGRAPH_STYLE = 0x10463

FITTING_ATTRS = (
    (0x6E83, "AutoFit", ("enum", ((0, "false"), (1, "true")))),
    (0x6E7E, "LeftCrop", ("number",)),
    (0x6E7F, "TopCrop", ("number",)),
    (0x6E80, "RightCrop", ("number",)),
    (0x6E81, "BottomCrop", ("number",)),
    (
        0x6E7C,
        "FittingOnEmptyFrame",
        ("enum", (
            (0, "None"),
            (1, "ContentToFrame"),
            (2, "Proportionally"),
            (3, "FillProportionally"),
        )),
    ),
    (
        0x6E7D,
        "FittingAlignment",
        ("enum", (
            (0, "TopLeftAnchor"),
            (1, "TopCenterAnchor"),
            (3, "LeftCenterAnchor"),
            (4, "CenterAnchor"),
        )),
    ),
)

ANCHORED_SETTINGS = (
    "AnchoredPosition",
    "SpineRelative",
    "PinPosition",
    "AnchorPoint",
    "HorizontalAlignment",
    "HorizontalReferencePoint",
    "VerticalAlignment",
    "VerticalReferencePoint",
    "AnchorXoffset",
    "AnchorYoffset",
)

CONTOUR_PARAMS = (
    ("ContourTolerance", "5"),
    ("ContourThreshold", "25"),
    ("ContourAlphaIndex", "0"),
    ("ContourPathIndex", "-1"),
)

CORNER_OPTIONS = (
    (0, "None"),
    (0x5A15, "RoundedCorner"),
    (0x5A16, "InverseRoundedCorner"),
    (0x5A17, "InsetCorner"),
    (0x5A18, "BevelCorner"),
    (0x5A19, "FancyCorner"),
)

STROKE_TYPES = (
    (0x5A29, "Solid"),
    (0x5A2A, "Dashed"),
    (0x5A37, "Canned Dashed 4x4"),
    (0x5A38, "Canned Dashed 3x2"),
    (0x5A39, "Canned Dotted"),
    (0x5A3A, "Wavy"),
    (0x5A3B, "Straight Hash"),
    (0x5A3C, "Right Slant Hash"),
    (0x5A3E, "White Diamond"),
    (0x5A3F, "Japanese Dots"),
    (0xB004, "ThinThin"),
    (0xB005, "ThinThick"),
    (0xB006, "ThickThin"),
    (0xB007, "ThickThick"),
    (0xB009, "ThickThinThick"),
    (0xB01A, "Triple_Stroke"),
)

BUILTIN_STROKE_STYLES = (
    "Triple_Stroke",
    "ThickThinThick",
    "ThinThickThin",
    "ThickThick",
    "ThickThin",
    "ThinThick",
    "ThinThin",
    "Japanese Dots",
    "White Diamond",
    "Left Slant Hash",
    "Right Slant Hash",
    "Straight Hash",
    "Wavy",
    "Canned Dotted",
    "Canned Dashed 3x2",
    "Canned Dashed 4x4",
    "Dashed",
    "Solid",
)

LINE_ENDS = (
    (0, "None"),
    (0x5A03, "SimpleArrowHead"),
    (0x5A04, "SimpleWideArrowHead"),
    (0x5A05, "TriangleArrowHead"),
    (0x5A06, "TriangleWideArrowHead"),
    (0x5A07, "BarbedArrowHead"),
    (0x5A08, "CurvedArrowHead"),
    (0x5A09, "CircleArrowHead"),
    (0x5A0A, "CircleSolidArrowHead"),
    (0x5A0C, "SquareSolidArrowHead"),
    (0x5A0D, "BarArrowHead"),
)

TEXT_ATTRS = (
    (0x1B01, "FillColor", ("swatch",), False),
    (0x1B02, "FontStyle", ("string",), False),
    (0x1B03, "PointSize", ("number",), False),
    (0x1B06, "HorizontalScale", ("percent",), False),
    (0x1B08, "Ligatures", ("equals", 1), False),
    (
        0x1B07,
        "KerningMethod",
        ("enum", (
            (15972, "$ID/Metrics"),
            (79875, "$ID/Optical"),
            (0x3E65, "$ID/Metrics - Roman Only"),
            (0x221, "$ID/None"),
        )),
        False,
    ),
    (0x1B0A, "StrokeWeight", ("number",), False),
    (0x1B0B, "Tracking", ("scale", 1000.0), False),
    (0x1B13, "KerningValue", ("kerning",), False),
    (
        0x1B0C,
        "Composer",
        ("enum", (
            (0x2001, "HL Single"),
            (0x2002, "HL Composer"),
            (0x2078, "HL Composer Optyca"),
            (0x2079, "HL Single Optyca"),
            (0x2010, "HL Composer J"),
            (0x2011, "HL Single J"),
        )),
        False,
    ),
    (0x1B0D, "DropCapCharacters", ("number",), False),
    (0x1B0E, "DropCapLines", ("number",), False),
    (0x1B10, "BaselineShift", ("number",), False),
    (
        0x1B11,
        "Capitalization",
        ("enum", (
            (0, "Normal"),
            (1, "SmallCaps"),
            (2, "AllCaps"),
            (3, "CapToSmallCap"),
            (4, "LowerCase"),
        )),
        False,
    ),
    (0x1B12, "StrokeColor", ("swatch",), False),
    (0x1B15, "VerticalScale", ("percent",), False),
    (0x1B16, "LeftIndent", ("number",), False),
    (0x1B17, "RightIndent", ("number",), False),
    (0x1B18, "FirstLineIndent", ("number",), False),
    (0x1B1A, "AutoLeading", ("percent",), False),
    (0x1B1B, "Leading", ("leading",), True),
    (0x1B1D, "AppliedLanguage", ("language",), False),
    (0x1B1F, "Hyphenation", ("equals", 3), False),
    (0x1B24, "NoBreak", ("equals", 1), False),
    (0x1B25, "HyphenationZone", ("number",), False),
    (0x1B26, "SpaceBefore", ("number",), False),
    (0x1B27, "SpaceAfter", ("number",), False),
    (0x1B29, "TabList", ("tab_list",), True),
    (0x1B2A, "Underline", ("equals", 1), False),
    (0x1B2B, "AppliedFont", ("font",), True),
    (0x42AE, "OpenTypeFeatures", ("features",), True),
    (
        0x1B2C,
        "OTFFigureStyle",
        ("enum", (
            (0, "TabularLining"),
            (1, "ProportionalOldstyle"),
            (2, "ProportionalLining"),
            (3, "TabularOldstyle"),
            (4, "Default"),
        )),
        False,
    ),
    (0x1B2E, "MaximumWordSpacing", ("percent",), False),
    (0x1B2F, "MinimumWordSpacing", ("percent",), False),
    (0x1B31, "MaximumLetterSpacing", ("percent",), False),
    (0x1B32, "MinimumLetterSpacing", ("percent",), False),
    (
        0x1B37,
        "StartParagraph",
        ("enum", (
            (0, "Anywhere"),
            (1, "NextColumn"),
            (2, "NextPage"),
            (3, "NextFrame"),
            (4, "NextOddPage"),
        )),
        False,
    ),
    (
        0x1B3C,
        "Position",
        ("enum", (
            (0, "Normal"),
            (1, "Superscript"),
            (2, "Subscript"),
            (3, "OTSuperscript"),
            (4, "OTSubscript"),
            (5, "OTNumerator"),
            (6, "OTDenominator"),
        )),
        False,
    ),
    (0x1B40, "KeepLinesTogether", ("equals", 1), False),
    (0x1B42, "FillTint", ("number",), False),
    (0x1B46, "GradientFillAngle", ("number",), False),
    (0x1B48, "GradientFillLength", ("number",), False),
    (0x1B4A, "GradientFillStart", ("point",), False),
    (0x1B4D, "RuleAboveLineWeight", ("number",), False),
    (0x1B4F, "RuleAboveOffset", ("number",), False),
    (0x1B50, "RuleAboveLeftIndent", ("number",), False),
    (0x1B51, "RuleAboveRightIndent", ("number",), False),
    (
        0x1B52,
        "RuleAboveWidth",
        ("enum", ((1, "ColumnWidth"), (2, "TextWidth"))),
        False,
    ),
    (0x1B53, "RuleBelowColor", ("swatch_or_text",), True),
    (0x1B54, "RuleBelowLineWeight", ("number",), False),
    (0x1B55, "RuleBelowTint", ("number",), False),
    (0x1B56, "RuleBelowOffset", ("number",), False),
    (0x1B5D, "RuleBelow", ("equals", 1), False),
    (
        0x1B6A,
        "ParagraphBreakType",
        ("enum", (
            (0, "Anywhere"),
            (1, "NextColumn"),
            (2, "NextPage"),
            (3, "NextFrame"),
            (4, "NextOddPage"),
        )),
        False,
    ),
    (
        0x1B6B,
        "SingleWordJustification",
        ("enum", ((0, "LeftAlign"), (3, "FullyJustified"))),
        False,
    ),
    (0x1B75, "AllNestedStyles", ("nested_styles",), True),
    (0x1BBA, "AllGREPStyles", ("grep_styles",), True),
    (0x1BBB, "AllLineStyles", ("line_styles",), True),
    (
        0x42C0,
        "TreatIdeographicSpaceAsSpace",
        ("equals", 1),
        False,
    ),
    (
        0x50F18,
        "DiacriticPosition",
        ("enum", ((4, "OpentypePosition"), (5, "OpentypePositionFromBaseline"))),
        False,
    ),
    (0x4221, "Mojikumi", ("cjk_set",), True),
    (0x4224, "KinsokuSet", ("cjk_set",), True),
    (
        0x1B7E,
        "Justification",
        ("enum", (
            (0, "LeftAlign"),
            (1, "CenterAlign"),
            (2, "RightAlign"),
            (3, "FullyJustified"),
            (4, "LeftJustified"),
            (5, "CenterJustified"),
            (6, "RightJustified"),
            (8, "ToBindingSide"),
            (9, "AwayFromBindingSide"),
        )),
        False,
    ),
    (0x1B80, "DropcapDetail", ("number",), False),
    (0x1B8C, "OTFContextualAlternate", ("equals", 1), False),
    (0x1B8D, "UnderlineColor", ("swatch_or_text",), True),
    (0x1B91, "UnderlineOffset", ("number",), False),
    (0x1B94, "UnderlineWeight", ("number",), False),
    (0x1BB7, "MiterLimit", ("number",), False),
    (
        0x1BB9,
        "EndJoin",
        ("enum", ((0, "MiterEndJoin"), (1, "RoundEndJoin"))),
        False,
    ),
    (
        0x1BBD,
        "SpanColumnType",
        ("enum", ((0, "SingleColumn"), (1, "SpanColumns"), (2, "SplitColumns"))),
        False,
    ),
    (
        0x1BBE,
        "SpanSplitColumnCount",
        ("number_or", 1.0, "All", "short"),
        True,
    ),
    (0x1BBF, "SplitColumnInsideGutter", ("number",), False),
    (0x1BC4, "SpanColumnMinSpaceAfter", ("number",), False),
    (0x1BD2, "ParagraphShadingColor", ("swatch",), True),
    (0x1BD3, "ParagraphShadingTint", ("number",), False),
    (0x1BD6, "ParagraphShadingOn", ("equals", 1), False),
    (0x1BDB, "ParagraphShadingTopOffset", ("number",), False),
    (0x1BDC, "ParagraphShadingBottomOffset", ("number",), False),
    (0x1BF6, "ParagraphBorderColor", ("swatch",), True),
    (0x1BFA, "ParagraphBorderGapColor", ("swatch_or_none",), True),
    (0x1BF9, "ParagraphBorderOn", ("equals", 1), False),
    (0x1DF03, "ParagraphBorderTopOffset", ("number",), False),
    (
        0x1DF12,
        "ParagraphShadingTopLeftCornerRadius",
        ("number",),
        False,
    ),
    (
        0x1DF14,
        "ParagraphShadingBottomLeftCornerRadius",
        ("number",),
        False,
    ),
    (0x1DF04, "ParagraphBorderBottomOffset", ("number",), False),
    (
        0x1DF21,
        "SameParaStyleSpacing",
        ("number_or", -1.0, "SetIgnore", "unit"),
        True,
    ),
    (0x4265, "GridAlignFirstLineOnly", ("equals", 1), False),
    (0x425E, "Tatechuyoko", ("equals", 1), False),
    (0x4279, "ShataiDegreeAngle", ("scale", 100.0), False),
    (0x427A, "ShataiAdjustTsume", ("equals", 1), False),
    (0x427B, "ShataiAdjustRotation", ("equals", 1), False),
    (0x422D, "RubyFlag", ("nonzero",), False),
    (0x422E, "RubyString", ("text",), False),
    (
        0x4266,
        "GridAlignment",
        ("enum", (
            (0, "None"),
            (1, "AlignBaseline"),
            (2, "AlignEmTop"),
            (3, "AlignEmCenter"),
        )),
        False,
    ),
    (
        0x1A401,
        "BulletsAndNumberingListType",
        ("enum", ((0, "NoList"), (1, "BulletList"), (2, "NumberedList"))),
        False,
    ),
    (0x1A406, "BulletChar", ("bullet_char",), True),
    (0x1A413, "BulletsFont", ("font_or_none",), True),
    (0x1A414, "BulletsFontStyle", ("string_or_nothing",), True),
    (0x1A417, "AppliedNumberingList", ("numbering_list",), True),
    (0x1A419, "NumberingContinue", ("equals", 1), False),
    (0x1A41F, "BulletsCharacterStyle", ("character_style",), True),
    (
        0x1A420,
        "NumberingCharacterStyle",
        ("character_style",),
        True,
    ),
    (0x1A423, "NumberingExpression", ("string",), False),


    (0x1B14, "HyphenateLadderLimit", ("number",), False),
    (0x1B68, "HyphenateLastWord", ("bool",), False),
    (0x1B85, "HyphenateAcrossColumns", ("bool",), False),
    (0x1B22, "HyphenateCapitalizedWords", ("bool",), False),
    (0x1B34, "MaximumGlyphScaling", ("percent",), False),
    (0x1B35, "MinimumGlyphScaling", ("percent",), False),
    (0x1B33, "DesiredGlyphScaling", ("percent",), False),
    (0x1B20, "HyphenateAfterFirst", ("number",), False),
    (0x1B23, "HyphenateWordsLongerThan", ("number",), False),
    (0x1B39, "KeepWithNext", ("number",), False),
    (0x1B2D, "DesiredWordSpacing", ("percent",), False),
    (0x1B21, "HyphenateBeforeLast", ("number",), False),
    (0x1B30, "DesiredLetterSpacing", ("percent",), False),
    (0x1B38, "KeepAllLinesTogether", ("bool",), False),
    (
        0x1B3E,
        "CharacterAlignment",
        ("enum", (
            (0, "AlignBaseline"),
            (1, "AlignEmCenter"),
            (2, "AlignEmBottom"),
            (3, "AlignEmTop"),
            (4, "AlignICFTop"),
            (5, "AlignICFBottom"),
        )),
        False,
    ),
    (
        0x1B09,
        "PageNumberType",
        ("enum", (
            (0, "AutoPageNumber"),
            (2, "NextPageNumber"),
            (3, "TextVariable"),
        )),
        False,
    ),
    (0x1B5E, "CustomGlyph", ("glyph_name",), True),
    (0x1B4C, "RuleAboveColor", ("swatch_or_text",), True),
    (0x1B5C, "RuleAbove", ("bool",), False),
    (0x1B83, "HyphenWeight", ("number",), False),
    (
        0x1B71,
        "RuleAboveType",
        ("builtin", "StrokeStyle/$ID/", STROKE_TYPES),
        True,
    ),
    (
        0x1B95,
        "UnderlineType",
        ("builtin", "StrokeStyle/$ID/", STROKE_TYPES),
        True,
    ),
    (
        0x1B72,
        "RuleBelowType",
        ("builtin", "StrokeStyle/$ID/", STROKE_TYPES),
        True,
    ),
    (
        0x1B9D,
        "StrikeThroughType",
        ("builtin", "StrokeStyle/$ID/", STROKE_TYPES),
        True,
    ),
    (
        0x1A40B,
        "NumberingFormat",
        ("enum_string", (
            (0x1A477, "1, 2, 3, 4..."),
            (0x1A478, "I, II, III, IV..."),
            (0x1A479, "i, ii, iii, iv..."),
            (0x1A47A, "A, B, C, D..."),
            (0x1A47B, "a, b, c, d..."),
            (0x1A497, "01,02,03..."),
            (0x1A4A7, "$ID/Circle Numbering"),
            (0, "None"),
        )),
        True,
    ),
    (
        0x1B73,
        "BalanceRaggedLines",
        ("enum", ((0, "NoBalancing"), (1, "VeeShape"))),
        True,
    ),
    (
        0x1A422,
        "NumberingAlignment",
        ("enum", ((0, "LeftAlign"), (2, "RightAlign"))),
        False,
    ),
    (
        0x1A421,
        "BulletsAlignment",
        ("enum", ((0, "LeftAlign"), (1, "CenterAlign"), (2, "RightAlign"))),
        False,
    ),
    (0x1B4E, "RuleAboveTint", ("number",), False),
    (
        0x50F29,
        "ParagraphDirection",
        ("enum", ((0, "LeftToRightDirection"), (1, "RightToLeftDirection"))),
        False,
    ),
    (
        0x50F2A,
        "ParagraphJustification",
        ("enum", (
            (0, "DefaultJustification"),
            (6, "NaskhKashidaJustificationFrac"),
        )),
        False,
    ),
    (
        0x50F1F,
        "CharacterDirection",
        ("enum", (
            (0, "DefaultDirection"),
            (1, "LeftToRightDirection"),
            (2, "RightToLeftDirection"),
        )),
        False,
    ),
    (
        0x50F11,
        "DigitsType",
        ("enum", ((0, "DefaultDigits"), (2, "HindiDigits"))),
        False,
    ),
    (0x1A418, "NumberingLevel", ("number",), False),
    (0x1A424, "BulletsTextAfter", ("string",), False),
    (0x1A41D, "NumberingApplyRestartPolicy", ("bool",), False),
    (
        0x1BB8,
        "StrokeAlignment",
        ("enum", ((0, "CenterAlignment"), (2, "OutsideAlignment"))),
        False,
    ),
    (
        0x4297,
        "LeadingModel",
        ("enum", (
            (0, "LeadingModelRoman"),
            (1, "LeadingModelAkiBelow"),
            (2, "LeadingModelAkiAbove"),
            (3, "LeadingModelCenter"),
        )),
        False,
    ),
    (0x1BC1, "KeepWithPrevious", ("bool",), False),
    (0x1B93, "UnderlineTint", ("number",), False),
    (0x1B8E, "UnderlineGapColor", ("swatch_or_text",), True),
    (0x1B77, "RuleAboveGapColor", ("swatch_or_text",), True),
    (0x1B96, "StrikeThroughColor", ("swatch_or_text",), True),
    (0x426C, "Rensuuji", ("bool",), False),
    (0x4295, "OTFProportionalMetrics", ("bool",), False),
    (0x1B57, "RuleBelowLeftIndent", ("number",), False),
    (0x1B58, "RuleBelowRightIndent", ("number",), False),
    (
        0x1B59,
        "RuleBelowWidth",
        ("enum", ((1, "ColumnWidth"), (2, "TextWidth"))),
        False,
    ),
    (0x1B86, "KeepRuleAboveInFrame", ("bool",), False),
    (0x1B6F, "OTFTitling", ("bool",), False),
    (0x1B6E, "OTFDiscretionaryLigature", ("bool",), False),
    (0x1BCA, "OTFSwash", ("bool",), False),
    (0x1B89, "OTFStylisticSets", ("number",), False),
    (0x1B44, "OverprintFill", ("bool",), False),
    (0x1B92, "UnderlineOverprint", ("bool",), False),
    (0x1B8F, "UnderlineGapOverprint", ("bool",), False),
    (0x1B3D, "StrikeThru", ("bool",), False),
    (0x1B9A, "StrikeThroughOffset", ("number",), False),
    (0x1B9E, "StrikeThroughWeight", ("number",), False),
    (0x1B9C, "StrikeThroughTint", ("number",), False),
    (0x1B78, "RuleAboveGapTint", ("number",), False),
    (0x1B4B, "Skew", ("number",), False),
    (0x1B65, "LastLineIndent", ("number",), False),
    (
        0x1B81,
        "PositionalForm",
        ("enum", ((0, "None"), (1, "Calculate"), (2, "Initial"))),
        False,
    ),
    (0x1B87, "IgnoreEdgeAlignment", ("bool",), False),
    (
        0x42AD,
        "GlyphForm",
        ("enum", (
            (0, "None"),
            (2, "ExpertForm"),
            (5, "MonospacedHalfWidthForm"),
            (9, "ProportionalWidthForm"),
            (10, "FullWidthForm"),
        )),
        False,
    ),
    (0x4222, "LeadingAki", ("number",), False),
    (0x4223, "TrailingAki", ("number",), False),
    (
        0x4225,
        "KinsokuType",
        ("enum", (
            (0, "KinsokuPushInFirst"),
            (2, "KinsokuPushOutOnly"),
            (3, "KinsokuPrioritizeAdjustmentAmount"),
        )),
        False,
    ),
    (
        0x4226,
        "KinsokuHangType",
        ("enum", (
            (0, "None"),
            (1, "KinsokuHangRegular"),
            (2, "KinsokuHangForce"),
        )),
        False,
    ),
    (0x4227, "BunriKinshi", ("bool",), False),
    (0x421D, "Tsume", ("number",), False),
    (0x42A6, "CjkGridTracking", ("bool",), False),
    (0x422C, "RubyOpenTypePro", ("bool",), False),
    (0x422F, "RubyFontSize", ("number",), False),
    (0x4248, "KentenFontSize", ("number",), False),
    (
        0x4232,
        "RubyType",
        ("enum", ((0, "GroupRuby"), (1, "PerCharacterRuby"))),
        False,
    ),
    (
        0x4247,
        "KentenKind",
        ("enum", (
            (0, "None"),
            (1, "KentenSesameDot"),
            (5, "KentenSmallBlackCircle"),
        )),
        False,
    ),

    (
        0x4231,
        "RubyAlignment",
        ("enum", ((1, "RubyCenter"), (2, "RubyRight"), (4, "RubyJIS"))),
        False,
    ),
    (
        0x4235,
        "RubyParentSpacing",
        ("enum", (
            (0, "RubyParentNoAdjustment"),
            (1, "RubyParentBothSides"),
            (2, "RubyParent121Aki"),
        )),
        False,
    ),
    (
        0x423C,
        "RubyParentOverhangAmount",
        ("enum", (
            (0, "None"),
            (1, "RubyOverhangOneRuby"),
            (5, "RubyOverhangNoLimit"),
        )),
        False,
    ),
    (
        0x423A,
        "RubyPosition",
        ("enum", ((0, "AboveRight"), (1, "BelowLeft"))),
        False,
    ),
    (0x4239, "RubyYOffset", ("number",), False),
    (0x423F, "RubyParentScalingPercent", ("percent",), False),
    (0x423E, "RubyAutoScaling", ("bool",), False),
    (0x423B, "RubyAutoAlign", ("bool",), False),
    (0x42B1, "RubyAutoTcyDigits", ("number",), False),
    (0x42B2, "RubyAutoTcyIncludeRoman", ("bool",), False),
    (0x4236, "RubyXScale", ("percent",), False),
    (0x424C, "KentenXScale", ("percent",), False),
    (
        0x4281,
        "WarichuAlignment",
        ("enum", ((0, "LeftAlign"), (7, "Auto"))),
        False,
    ),
    (0x427D, "Warichu", ("bool",), False),
    (0x427E, "WarichuLines", ("number",), False),
    (0x427F, "WarichuSize", ("percent",), False),
    (0x4280, "WarichuLineSpacing", ("number",), False),
    (0x4234, "RubyFontStyle", ("empty_as_nothing",), True),
    (0x424B, "KentenFontStyle", ("empty_as_nothing",), True),
    (0x1DF1F, "MergeConsecutiveParaBorders", ("bool",), False),
    (0x1DF1A, "ParagraphBorderTopLineWeight", ("number",), False),
    (
        0x1DF1B,
        "ParagraphBorderBottomLineWeight",
        ("number",),
        False,
    ),
    (
        0x1DF1C,
        "ParagraphBorderLeftLineWeight",
        ("number",),
        False,
    ),
    (
        0x1DF1D,
        "ParagraphBorderRightLineWeight",
        ("number",),
        False,
    ),
    (0x1DF01, "ParagraphBorderLeftOffset", ("number",), False),
    (0x1DF02, "ParagraphBorderRightOffset", ("number",), False),
    (0x1BF7, "ParagraphBorderTint", ("number",), False),
    (
        0x1DF16,
        "ParagraphBorderStrokeEndCap",
        ("enum", ((0, "ButtEndCap"), (1, "RoundEndCap"))),
        False,
    ),
    (
        0x1DF17,
        "ParagraphBorderWidth",
        ("enum", ((0, "ColumnWidth"), (1, "TextWidth"))),
        False,
    ),
    (
        0x1DF20,
        "ProviderHyphenationStyle",
        ("enum", ((0, "HyphAll"), (3, "HyphPreferredAesthetic"))),
        False,
    ),
    (0x1BD9, "ParagraphShadingLeftOffset", ("number",), False),
    (0x1BDA, "ParagraphShadingRightOffset", ("number",), False),
    (
        0x1BD5,
        "ParagraphShadingWidth",
        ("enum", ((0, "ColumnWidth"), (1, "TextWidth"))),
        False,
    ),
    (0x1BD7, "ParagraphShadingClipToFrame", ("bool",), False),
    (0x1BD4, "ParagraphShadingOverprint", ("bool",), False),
    (
        0x1BFD,
        "ParagraphBorderType",
        ("builtin", "StrokeStyle/$ID/", STROKE_TYPES),
        True,
    ),
    (0x1B3A, "KeepFirstLines", ("number",), False),
    (0x1B3B, "KeepLastLines", ("number",), False),
    (0x42C1, "AllowArbitraryHyphenation", ("bool",), False),
    (0x4264, "GridGyoudori", ("number",), False),
    (0x4268, "RotateSingleByteCharacters", ("bool",), False),
    (0x4261, "AutoTcy", ("number",), False),
    (0x42A3, "ScaleAffectsLineHeight", ("bool",), False),
    (0x4233, "RubyFont", ("font_or_none",), True),
    (0x4257, "KentenFillColor", ("swatch_or_text",), True),


    (0x1B41, "StrokeTint", ("number",), False),
    (0x1B43, "OverprintStroke", ("bool",), False),
    (0x1B45, "GradientStrokeAngle", ("number",), False),
    (0x1B47, "GradientStrokeLength", ("number",), False),
    (0x1B49, "GradientStrokeStart", ("point",), False),
    (0x1B5A, "RuleAboveOverprint", ("bool",), False),
    (0x1B5B, "RuleBelowOverprint", ("bool",), False),
    (0x1B6C, "OTFOrdinal", ("bool",), False),
    (0x1B6D, "OTFFraction", ("bool",), False),
    (0x1B79, "RuleAboveGapOverprint", ("bool",), False),
    (0x1B7B, "RuleBelowGapColor", ("swatch_or_text",), True),
    (0x1B7C, "RuleBelowGapTint", ("number",), False),
    (0x1B7D, "RuleBelowGapOverprint", ("bool",), False),
    (0x1B82, "OTFMark", ("bool",), False),
    (0x1B84, "OTFLocale", ("bool",), False),
    (0x1B88, "OTFSlashedZero", ("bool",), False),
    (0x1B8A, "OTFHistorical", ("bool",), False),
    (0x1B90, "UnderlineGapTint", ("number",), False),
    (0x1B97, "StrikeThroughGapColor", ("swatch_or_text",), True),
    (0x1B98, "StrikeThroughGapOverprint", ("bool",), False),
    (0x1B99, "StrikeThroughGapTint", ("number",), False),
    (0x1B9B, "StrikeThroughOverprint", ("bool",), False),
    (0x1BC0, "SplitColumnOutsideGutter", ("number",), False),
    (0x1BC3, "SpanColumnMinSpaceBefore", ("number",), False),
    (
        0x1BD8,
        "ParagraphShadingSuppressPrinting",
        ("bool",),
        False,
    ),
    (0x1BF8, "ParagraphBorderOverprint", ("bool",), False),
    (0x1BFB, "ParagraphBorderGapTint", ("number",), False),
    (0x1BFC, "ParagraphBorderGapOverprint", ("bool",), False),
    (0x4237, "RubyYScale", ("percent",), False),
    (0x4238, "RubyXOffset", ("number",), False),
    (0x423D, "RubyOverhang", ("bool",), False),
    (0x4240, "RubyFill", ("swatch_or_text",), True),
    (0x4241, "RubyTint", ("number",), False),
    (
        0x4242,
        "RubyOverprintFill",
        ("enum", ((0xFFFF, "Auto"),)),
        False,
    ),
    (0x4243, "RubyStroke", ("swatch_or_text",), True),
    (0x4244, "RubyStrokeTint", ("number",), False),
    (
        0x4245,
        "RubyOverprintStroke",
        ("enum", ((0xFFFF, "Auto"),)),
        False,
    ),
    (0x4246, "RubyWeight", ("number",), False),
    (0x424A, "KentenFont", ("font_or_none",), True),
    (0x424D, "KentenYScale", ("percent",), False),
    (0x424F, "KentenPlacement", ("number",), False),
    (
        0x4250,
        "KentenAlignment",
        ("enum", ((1, "AlignKentenCenter"),)),
        False,
    ),
    (
        0x4251,
        "KentenPosition",
        ("enum", ((0, "AboveRight"),)),
        False,
    ),
    (
        0x4252,
        "KentenCustomCharacter",
        ("enum", ((0, ""),)),
        False,
    ),
    (
        0x4253,
        "KentenCharacterSet",
        ("enum", ((0, "CharacterInput"),)),
        False,
    ),
    (0x4258, "KentenTint", ("number",), False),
    (
        0x4259,
        "KentenOverprintFill",
        ("enum", ((0xFFFF, "Auto"),)),
        False,
    ),
    (0x425A, "KentenStrokeColor", ("swatch_or_text",), True),
    (0x425B, "KentenStrokeTint", ("number",), False),
    (
        0x425C,
        "KentenOverprintStroke",
        ("enum", ((0xFFFF, "Auto"),)),
        False,
    ),
    (0x425D, "KentenWeight", ("number",), False),
    (0x425F, "TatechuyokoXOffset", ("number",), False),
    (0x4260, "TatechuyokoYOffset", ("number",), False),
    (0x4262, "AutoTcyIncludeRoman", ("bool",), False),
    (0x4263, "Jidori", ("number",), False),
    (0x4267, "CharacterRotation", ("number",), False),
    (0x4278, "ShataiMagnification", ("number",), False),
    (0x428E, "WarichuCharsBeforeBreak", ("number",), False),
    (0x428F, "WarichuCharsAfterBreak", ("number",), False),
    (0x4294, "OTFHVKana", ("bool",), False),
    (0x4296, "OTFRomanItalics", ("bool",), False),
    (0x42A4, "ParagraphGyoudori", ("bool",), False),
    (0x42B3, "RubyAutoTcyAutoScale", ("bool",), False),
    (0x1A407, "NumberingStartAt", ("number",), False),
    (
        0x1A41E,
        "NumberingRestartPolicies",
        ("restart_policies",),
        True,
    ),
    (
        0x1DF05,
        "ParagraphBorderStrokeEndJoin",
        ("enum", ((0, "MiterEndJoin"),)),
        False,
    ),
    (
        0x1DF06,
        "ParagraphBorderTopLeftCornerOption",
        ("enum", ((0, "None"),)),
        False,
    ),
    (
        0x1DF07,
        "ParagraphBorderTopRightCornerOption",
        ("enum", ((0, "None"),)),
        False,
    ),
    (
        0x1DF08,
        "ParagraphBorderBottomLeftCornerOption",
        ("enum", ((0, "None"),)),
        False,
    ),
    (
        0x1DF09,
        "ParagraphBorderBottomRightCornerOption",
        ("enum", ((0, "None"),)),
        False,
    ),
    (0x1DF1E, "ParagraphBorderDisplayIfSplits", ("bool",), False),
    (
        0x50F15,
        "Kashidas",
        ("enum", ((0, "DefaultKashidas"),)),
        False,
    ),
    (0x50F2B, "ParagraphKashidaWidth", ("number",), False),
    (0x50F2F, "XOffsetDiacritic", ("number",), False),
    (0x50F30, "YOffsetDiacritic", ("number",), False),
    (0x50F3C, "OTFOverlapSwash", ("bool",), False),
    (0x50F3D, "OTFStylisticAlternate", ("bool",), False),
    (0x50F3E, "OTFJustificationAlternate", ("bool",), False),
    (0x50F3F, "OTFStretchedAlternate", ("bool",), False),
    (
        0x50F40,
        "KeyboardDirection",
        ("enum", ((0, "DefaultDirection"), (1, "LeftToRightDirection"))),
        False,
    ),
)

SHADING_CORNERS = (
    (0, "None"),
    (0x5A15, "RoundedCorner"),
    (0x5A18, "BevelCorner"),
)

TIED_TEXT_ATTRS = (
    (
        (0x1DF0A, 0x1DF0B, 0x1DF0C, 0x1DF0D),
        (
            "ParagraphBorderTopLeftCornerRadius",
            "ParagraphBorderTopRightCornerRadius",
            "ParagraphBorderBottomLeftCornerRadius",
            "ParagraphBorderBottomRightCornerRadius",
        ),
        ("number",),
    ),
    (
        (0x1DF13, 0x1DF15),
        (
            "ParagraphShadingTopRightCornerRadius",
            "ParagraphShadingBottomRightCornerRadius",
        ),
        ("number",),
    ),
    (
        (0x1DF0E, 0x1DF10),
        (
            "ParagraphShadingTopLeftCornerOption",
            "ParagraphShadingBottomLeftCornerOption",
        ),
        ("enum", SHADING_CORNERS),
    ),
    (
        (0x1DF0F, 0x1DF11),
        (
            "ParagraphShadingTopRightCornerOption",
            "ParagraphShadingBottomRightCornerOption",
        ),
        ("enum", SHADING_CORNERS),
    ),
)

PAIRED_TEXT_ATTRS = (
    (
        (0x1BDD, 0x1BDE),
        ("ParagraphShadingTopOrigin", "ParagraphShadingBottomOrigin"),
        (
            ((0, 0), ("AscentTopOrigin", "DescentBottomOrigin")),
            ((3, 2), ("EmBoxTopOrigin", "EmBoxBottomOrigin")),
            ((1, 1), ("BaselineTopOrigin", "BaselineBottomOrigin")),
        ),
    ),
    (
        (0x1DF18, 0x1DF19),
        ("ParagraphBorderTopOrigin", "ParagraphBorderBottomOrigin"),
        (
            ((0, 0), ("AscentTopOrigin", "DescentBottomOrigin")),
            ((3, 2), ("EmBoxTopOrigin", "EmBoxBottomOrigin")),
        ),
    ),
)

ITEM_ATTRS = (
    (0x6E68, "FillColor", ("swatch",)),
    (0x6E69, "FillTint", ("tint",)),
    (0x6E64, "StrokeColor", ("swatch",)),
    (0x6E65, "StrokeWeight", ("number",)),
    (0x6E6D, "MiterLimit", ("number",)),
    (0x6E6F, "CornerOption", ("enum", CORNER_OPTIONS)),
    (0x6E70, "CornerRadius", ("number",)),
    (0x6E70, "TopLeftCornerRadius", ("number",)),
    (0x6E94, "TopRightCornerRadius", ("number",)),
    (0x6E92, "BottomLeftCornerRadius", ("number",)),
    (0x6E93, "BottomRightCornerRadius", ("number",)),
    (0x6E6F, "TopLeftCornerOption", ("enum", CORNER_OPTIONS)),
    (0x6E91, "TopRightCornerOption", ("enum", CORNER_OPTIONS)),
    (0x6E8F, "BottomLeftCornerOption", ("enum", CORNER_OPTIONS)),
    (
        0x6E90,
        "BottomRightCornerOption",
        ("enum", CORNER_OPTIONS),
    ),
    (0x551F, "GradientFillLength", ("number",)),
    (0x5520, "GradientFillStart", ("point",)),
    (0x5525, "GradientStrokeLength", ("number",)),
    (0x5526, "GradientStrokeStart", ("point",)),
    (
        0x6E6E,
        "StrokeType",
        ("builtin", "StrokeStyle/$ID/", STROKE_TYPES),
    ),
    (
        0x6E8C,
        "StrokeAlignment",
        ("enum", (
            (0, "CenterAlignment"),
            (1, "InsideAlignment"),
            (2, "OutsideAlignment"),
        )),
    ),
    (0x6E66, "StrokeTint", ("tint",)),
    (0x5A35, "StrokeDashAndGap", ("dashes",)),
    (0x5A35, "StrokeCornerAdjustment", ("dash_corner",)),
    (0x6E89, "GapColor", ("swatch",)),
    (0x6E8A, "GapTint", ("tint",)),
    (
        0x6E6B,
        "EndCap",
        ("enum", (
            (0, "ButtEndCap"),
            (1, "RoundEndCap"),
            (2, "ProjectingEndCap"),
        )),
    ),
    (
        0x6E6C,
        "EndJoin",
        ("enum", (
            (0, "MiterEndJoin"),
            (1, "RoundEndJoin"),
            (2, "BevelEndJoin"),
        )),
    ),
    (0x6E71, "LeftLineEnd", ("enum", LINE_ENDS)),
    (0x6E72, "RightLineEnd", ("enum", LINE_ENDS)),
    (
        0x6E84,
        "ArrowHeadAlignment",
        ("enum", ((0, "InsidePath"), (1, "OutsidePath"))),
    ),
)

ITEM_DEFAULT_ATTRS = (
    (0x551E, "GradientFillAngle", ("number",)),
    (0x5524, "GradientStrokeAngle", ("number",)),
    (0x6E78, "Nonprinting", ("bool",)),
    (0x6E95, "LeftArrowHeadScale", ("number",)),
    (0x6E96, "RightArrowHeadScale", ("number",)),
)

GRADIENT_ATTRS = (
    (0x5520, "GradientFillStart", "0 0"),
    (0x551F, "GradientFillLength", "0"),
    (0x551E, "GradientFillAngle", "0"),
    (0x5526, "GradientStrokeStart", "0 0"),
    (0x5525, "GradientStrokeLength", "0"),
    (0x5524, "GradientStrokeAngle", "0"),
    (0x5522, "GradientFillHiliteLength", "0"),
    (0x5523, "GradientFillHiliteAngle", "0"),
    (0x5528, "GradientStrokeHiliteLength", "0"),
    (0x5529, "GradientStrokeHiliteAngle", "0"),
)

CELL_EDGE_PRIORITY = (0xB6F9, 0xB6FA, 0xB6FB, 0xB6FC)

CELL_STYLE_EDGE = (2, 3, 1, 0)

CELL_ATTRS = (
    (0xB62C, ("TextTopInset", "TopInset"), ("number",)),
    (0xB62B, ("TextLeftInset", "LeftInset"), ("number",)),
    (0xB62E, ("TextBottomInset", "BottomInset"), ("number",)),
    (0xB62D, ("TextRightInset", "RightInset"), ("number",)),
    (0x10470, ("GraphicLeftInset"), ("number",)),
    (0x10471, ("GraphicTopInset"), ("number",)),
    (0x10472, ("GraphicRightInset"), ("number",)),
    (0x10473, ("GraphicBottomInset"), ("number",)),
    (0xB63D, ("FillColor"), ("swatch",)),
    (0xB63E, ("FillTint"), ("number",)),
    (0xB639, ("OverprintFill"), ("enum", ((0, "false"),))),
    (
        0xB677,
        ("VerticalJustification"),
        ("enum", (
            (0, "TopAlign"),
            (1, "CenterAlign"),
            (2, "BottomAlign"),
            (3, "JustifyAlign"),
        )),
    ),
    (
        0xB676,
        ("FirstBaselineOffset"),
        ("enum", (
            (0, "LeadingOffset"),
            (1, "AscentOffset"),
            (3, "EmboxHeight"),
        )),
    ),
    (0xB6E1, ("WritingDirection"), ("enum", ((1, "true"),))),
    (0xB675, ("RotationAngle"), ("number",)),
    (0xB6DE, ("ClipContentToCell"), ("enum", ((0, "false"),))),
    (
        0xB6DC,
        ("DiagonalLineStrokeOverprint"),
        ("enum", ((0, "false"),)),
    ),
)

CELL_EDGE_ATTRS = (
    (
        (0xB645, 0xB646, 0xB647, 0xB648),
        (
            "LeftEdgeStrokeWeight",
            "RightEdgeStrokeWeight",
            "TopEdgeStrokeWeight",
            "BottomEdgeStrokeWeight",
        ),
        ("number",),
    ),
    (
        (0xB64D, 0xB64E, 0xB64F, 0xB650),
        (
            "LeftEdgeStrokeType",
            "RightEdgeStrokeType",
            "TopEdgeStrokeType",
            "BottomEdgeStrokeType",
        ),
        ("stroke_type",),
    ),
    (
        (0xB649, 0xB64B, 0xB64A, 0xB64C),
        (
            "LeftEdgeStrokeColor",
            "RightEdgeStrokeColor",
            "TopEdgeStrokeColor",
            "BottomEdgeStrokeColor",
        ),
        ("swatch",),
    ),
    (
        (0xB6A8, 0xB6A9, 0xB6AA, 0xB6AB),
        (
            "LeftEdgeStrokeTint",
            "RightEdgeStrokeTint",
            "TopEdgeStrokeTint",
            "BottomEdgeStrokeTint",
        ),
        ("edge_tint",),
    ),
    (
        CELL_EDGE_PRIORITY,
        (
            "LeftEdgeStrokePriority",
            "RightEdgeStrokePriority",
            "TopEdgeStrokePriority",
            "BottomEdgeStrokePriority",
        ),
        ("integer",),
    ),
    (
        (0x1040F, 0x10410, 0x10411, 0x10412),
        (
            "LeftEdgeStrokeGapTint",
            "RightEdgeStrokeGapTint",
            "TopEdgeStrokeGapTint",
            "BottomEdgeStrokeGapTint",
        ),
        ("edge_tint",),
    ),
    (
        (0x10420, 0x10421, 0x10422, 0x10423),
        (
            "LeftEdgeStrokeGapColor",
            "RightEdgeStrokeGapColor",
            "TopEdgeStrokeGapColor",
            "BottomEdgeStrokeGapColor",
        ),
        ("swatch",),
    ),
    (
        (0xB6BA, 0xB6BB, 0xB6BC, 0xB6BD),
        (
            "LeftEdgeStrokeOverprint",
            "RightEdgeStrokeOverprint",
            "TopEdgeStrokeOverprint",
            "BottomEdgeStrokeOverprint",
        ),
        ("enum", ((0, "false"),)),
    ),
    (
        (0x10431, 0x10432, 0x10433, 0x10434),
        (
            "LeftEdgeStrokeGapOverprint",
            "RightEdgeStrokeGapOverprint",
            "TopEdgeStrokeGapOverprint",
            "BottomEdgeStrokeGapOverprint",
        ),
        ("enum", ((0, "false"),)),
    ),
)

TABLE_ATTRS = (
    (0xB662, "SpaceBefore", ("number",)),
    (0xB663, "SpaceAfter", ("number",)),
    (0xB686, "StartRowStrokeCount", ("integer",)),
    (0xB687, "EndRowStrokeCount", ("integer",)),
    (0xB68C, "StartColumnStrokeCount", ("integer",)),
    (0xB68D, "EndColumnStrokeCount", ("integer",)),
    (0xB67D, "StartRowFillCount", ("integer",)),
    (0xB67E, "EndRowFillCount", ("integer",)),
    (0xB684, "StartRowStrokeColor", ("swatch",)),
    (0xB685, "EndRowStrokeColor", ("swatch",)),
    (0xB68A, "StartColumnStrokeColor", ("swatch",)),
    (0xB68B, "EndColumnStrokeColor", ("swatch",)),
    (0xB690, "StartRowStrokeWeight", ("number",)),
    (0xB691, "EndRowStrokeWeight", ("number",)),
    (0xB692, "StartColumnStrokeWeight", ("number",)),
    (0xB693, "EndColumnStrokeWeight", ("number",)),
    (0xB688, "StartRowStrokeType", ("stroke_type",)),
    (0xB689, "EndRowStrokeType", ("stroke_type",)),
    (0xB6B4, "StartRowStrokeTint", ("number",)),
    (0xB6B5, "EndRowStrokeTint", ("number",)),
    (0xB6B6, "StartColumnStrokeTint", ("number",)),
    (0xB6B7, "EndColumnStrokeTint", ("number",)),
    (0xB683, "ColumnFillsPriority", ("enum", ((0, "false"),))),
    (0xB67B, "StartRowFillColor", ("swatch",)),
    (0xB67C, "EndRowFillColor", ("swatch",)),
    (0xB67F, "StartColumnFillColor", ("swatch",)),
    (0xB680, "EndColumnFillColor", ("swatch",)),
    (0xB6B0, "StartRowFillTint", ("number",)),
    (0xB6B1, "EndRowFillTint", ("number",)),
    (0xB6B2, "StartColumnFillTint", ("number",)),
    (0xB6B3, "EndColumnFillTint", ("number",)),
    (0xB695, "SkipFirstAlternatingFillRows", ("integer",)),
    (0xB6E2, "SkipFirstAlternatingStrokeRows", ("integer",)),
)

GRAPHIC_CELL_ATTRS = (
    (0x10470, "GraphicLeftInset", ("number",)),
    (0x10471, "GraphicTopInset", ("number",)),
    (0x10472, "GraphicRightInset", ("number",)),
    (0x10473, "GraphicBottomInset", ("number",)),
    (
        0x10478,
        "ClipContentToGraphicCell",
        ("enum", ((0, "false"),)),
    ),
)

TABLE_ONLY_ATTRS = (
    (0xB62B, "LeftInset", ("number",)),
    (0xB62C, "TopInset", ("number",)),
    (0xB62D, "RightInset", ("number",)),
    (0xB62E, "BottomInset", ("number",)),
    (0x10408, "BreakHeaders", ("enum", ((2, "OncePerPage"),))),
)

TABLE_STYLE_ONLY_ATTRS = (
    (0x10450, "HeaderRegionCellStyle", ("cell_style",)),
    (0x10451, "FooterRegionCellStyle", ("cell_style",)),
    (0x10452, "BodyRegionCellStyle", ("cell_style",)),
    (0x10453, "LeftColumnRegionCellStyle", ("cell_style",)),
    (0x10454, "RightColumnRegionCellStyle", ("cell_style",)),
    (
        0x10457,
        "HeaderRegionSameAsBodyRegion",
        ("enum", ((0, "false"), (1, "true"))),
    ),
    (
        0x10458,
        "FooterRegionSameAsBodyRegion",
        ("enum", ((0, "false"), (1, "true"))),
    ),
    (
        0x10459,
        "LeftColumnRegionSameAsBodyRegion",
        ("enum", ((0, "false"), (1, "true"))),
    ),
    (
        0x1045A,
        "RightColumnRegionSameAsBodyRegion",
        ("enum", ((0, "false"), (1, "true"))),
    ),
)

TABLE_GROUPS = (
    (
        (0xB653, 0xB656, 0xB659, 0xB65C),
        (
            "TopBorderStrokeWeight",
            "LeftBorderStrokeWeight",
            "BottomBorderStrokeWeight",
            "RightBorderStrokeWeight",
        ),
        ("number",),
        False,
    ),
    (
        (0xB654, 0xB657, 0xB65A, 0xB65D),
        (
            "TopBorderStrokeColor",
            "LeftBorderStrokeColor",
            "BottomBorderStrokeColor",
            "RightBorderStrokeColor",
        ),
        ("swatch",),
        False,
    ),
    (
        (0xB655, 0xB658, 0xB65B, 0xB65E),
        (
            "TopBorderStrokeType",
            "LeftBorderStrokeType",
            "BottomBorderStrokeType",
            "RightBorderStrokeType",
        ),
        ("stroke_type",),
        False,
    ),
    (
        (0xB6AC, 0xB6AD, 0xB6AE, 0xB6AF),
        (
            "TopBorderStrokeTint",
            "LeftBorderStrokeTint",
            "BottomBorderStrokeTint",
            "RightBorderStrokeTint",
        ),
        ("number",),
        False,
    ),
    (
        (0x10429, 0x1042A, 0x1042B, 0x1042C),
        (
            "TopBorderStrokeGapColor",
            "LeftBorderStrokeGapColor",
            "BottomBorderStrokeGapColor",
            "RightBorderStrokeGapColor",
        ),
        ("swatch",),
        False,
    ),
    (
        (0xB68E, 0xB68F),
        ("StartColumnStrokeType", "EndColumnLineStyle"),
        ("stroke_type",),
        False,
    ),
    (
        (0x1040A, 0x1040B),
        ("SkipFirstHeader", "SkipLastFooter"),
        ("enum", ((1, "true"),)),
        True,
    ),
)

TEXT_CELL_ATTRS = (
    (0xB62C, "TextTopInset", ("number",)),
    (0xB62B, "TextLeftInset", ("number",)),
    (0xB62E, "TextBottomInset", ("number",)),
    (0xB62D, "TextRightInset", ("number",)),
    (0xB6DE, "ClipContentToTextCell", ("bool",)),
)

ROW_ATTRS = (
    (0xB69F, "AutoGrow", ("bool",)),
    (
        0x10407,
        "StartRow",
        ("enum", ((0, "Anywhere"), (2, "NextColumn"))),
    ),
    (0xB6A1, "KeepWithNextRow", ("enum", ((0, "false"),))),
)



# --- helpers -----------------------------------------------------------

def builtin_cjk_table(name):
    """The IDML enumeration value of a built-in kinsoku or mojikumi
    table."""
    tables = [
        ("kHardKinsokuName", "HardKinsoku"),
        ("kSoftKinsokuName", "SoftKinsoku"),
        ("kKoreanKinsokuName", "KoreanKinsoku"),
        ("kSimpChineseKinsokuName", "SimplifiedChineseKinsoku"),
        ("kTradChineseKinsokuName", "TraditionalChineseKinsoku"),
        ("kMojikumiDefaultName1", "LineEndAllOneHalfEmEnum"),
        ("kMojikumiDefaultName2", "OneEmIndentLineEndUkeOneHalfEmEnum"),
        ("kMojikumiDefaultName10", "LineEndAllOneEmEnum"),
        ("kMojikumiDefaultName14", "LineEndPeriodOneEmEnum"),
        ("kMojikumiDefaultName15", "TradChineseDefault"),
        ("kMojikumiDefaultName16", "SimpChineseDefault"),
    ]
    for k, e in tables:
        if k == name:
            return e
    return None


def custom_stroke_ref(st):
    """The ``Self`` of a custom stroke style."""
    kind = "DashedStrokeStyle" if st.dashed else "StripedStrokeStyle"
    return f"{kind}/{self_name(st.name)}"


def anchored_settings(a):
    """IDML attributes of anchored object settings, in IDML order, for
    the codes the samples show."""
    def code(v, codes):
        if v is None:
            return None
        for c, s in codes:
            if c == v:
                return s
        return None

    point = {
        (0, 2): "BottomRightAnchor",
        (2, 2): "BottomLeftAnchor",
        (2, 0): "TopLeftAnchor",
        (0, 0): "TopRightAnchor",
        (2, 1): "LeftCenterAnchor",
    }.get(a.point)
    bool_codes = [(0, "false"), (1, "true")]
    named = [
        ("AnchoredPosition", code(a.position, [(0, "InlinePosition"), (1, "Anchored"), (2, "AboveLine")])),
        ("SpineRelative", code(a.spine_relative, bool_codes)),
        ("PinPosition", code(a.pin, bool_codes)),
        ("AnchorPoint", point),
        ("HorizontalAlignment", code(a.horizontal_alignment, [(0, "RightAlign"), (1, "CenterAlign"), (2, "LeftAlign")])),
        ("HorizontalReferencePoint", code(a.horizontal_reference, [(1, "TextFrame"), (2, "PageMargins"), (3, "PageEdge"), (4, "AnchorLocation")])),
        ("VerticalAlignment", code(a.vertical_alignment, [(0, "TopAlign"), (1, "CenterAlign"), (2, "BottomAlign")])),
        ("VerticalReferencePoint", code(a.vertical_reference, [(2, "PageMargins"), (3, "PageEdge"), (4, "LineBaseline"), (6, "Capheight")])),
    ]
    out = [(k, v) for k, v in named if v is not None]
    if a.x_offset is not None:
        out.append(("AnchorXoffset", num(a.x_offset)))
    if a.y_offset is not None:
        out.append(("AnchorYoffset", num(a.y_offset)))
    return out


def fitting_attrs(w, attrs, ids):
    """Frame fitting attributes of ``attrs`` with IDs in ``ids``, in
    IDML order."""
    out = []
    for attr_id, name, kind in FITTING_ATTRS:
        if attr_id not in ids:
            continue
        v = attrs.get(attr_id)
        if v is None:
            continue
        text = value_text(w, kind, v)
        if text is not None:
            out.append((name, text))
    return out


def text_wrap_mode(wrap):
    """IDML ``TextWrapMode`` of a text wrap; no wrap is ``None``.
    ``None`` for a mode without evidence."""
    from .model import WrapMode
    mode = None if wrap is None else wrap.mode
    if mode is None or mode == WrapMode.NONE:
        return "None"
    if mode == WrapMode.JUMP_OBJECT:
        return "JumpObjectTextWrap"
    if mode == WrapMode.BOUNDING_BOX:
        return "BoundingBoxTextWrap"
    if mode == WrapMode.CONTOUR:
        return "Contour"
    return None


def text_wrap_side(code):
    """IDML ``TextWrapSide`` of the side code of a text wrap; ``None``
    for a code without evidence."""
    return {
        0: "BothSides", 1: "LeftSide", 2: "RightSide",
        3: "SideTowardsSpine", 4: "SideAwayFromSpine", 5: "LargestArea",
    }.get(code)


def contour_type(code):
    """IDML ``ContourType`` of a contour code (``Contour.kind``);
    ``None`` for a code not decoded."""
    return {
        0: "BoundingBox", 1: "DetectEdges", 2: "AlphaChannel",
        3: "PhotoshopPath", 4: "GraphicFrame", 5: "SameAsClipping",
        6: "SelectSubject",
    }.get(code)


def text_wrap_offsets(wrap):
    """IDML ``TextWrapOffset`` attributes of a text wrap (0 without
    one)."""
    left, top, right, bottom = (0.0, 0.0, 0.0, 0.0) if wrap is None else wrap.offsets
    return [("Top", num(top)), ("Left", num(left)),
            ("Bottom", num(bottom)), ("Right", num(right))]


def dash_written(name, stroke_type):
    """Whether a dash attribute (``StrokeDashAndGap``,
    ``StrokeCornerAdjustment``) is written: IDML has them only where
    the stroke type in effect is ``Dashed``."""
    return name not in ("StrokeDashAndGap", "StrokeCornerAdjustment") \
        or stroke_type == "StrokeStyle/$ID/Dashed"


def end_cap(code):
    """IDML ``EndCap`` of an end cap code (attribute 0x6E6B)."""
    return {0: "ButtEndCap", 1: "RoundEndCap", 2: "ProjectingEndCap"}.get(code)


def layout_constraints(flags):
    """IDML ``HorizontalLayoutConstraints`` and
    ``VerticalLayoutConstraints`` of layout constraint flags (chunk
    0x22228): bits 4 to 6 and 0 to 2, each set for
    ``FixedDimension``. ``None`` if another bit is set."""
    if flags & 0x88:
        return None

    def side(bits):
        return " ".join("FixedDimension" if bits & (1 << i) else "FlexibleDimension"
                        for i in range(3))

    return (side(flags >> 4), side(flags & 7))


def join_numbers(v):
    """Numbers separated by spaces."""
    return " ".join(str(u) for u in v)


def tab_list(stops):
    """Records of a ``TabList``. ``None`` for an alignment code without
    evidence."""
    out = []
    for t in stops:
        alignment = {0: "LeftAlign", 1: "CenterAlign", 2: "RightAlign",
                     3: "CharacterAlign"}.get(t.alignment)
        if alignment is None:
            return None
        # Only stops aligned on a character store one; IDML writes `.`
        # for the others.
        character = t.character if t.character is not None else "."
        out.append([
            ("Alignment", "enumeration", alignment),
            ("AlignmentCharacter", "string", character),
            ("Leader", "string", t.leader),
            ("Position", "unit", num(t.position)),
        ])
    return out


def bullet_char(kind, value):
    """``BulletChar`` attributes."""
    kind_name = {0: "UnicodeOnly", 1: "UnicodeWithFont",
                 2: "GlyphWithFont"}.get(kind)
    if kind_name is None:
        return None
    return ("attrs", [("BulletCharacterType", kind_name),
                      ("BulletCharacterValue", str(value))])


def restart_policies(words):
    """``NumberingRestartPolicies`` attributes of the two stored word
    triples seen in the corpus."""
    if tuple(words) == (0, 0, 0):
        policy, level = "AnyPreviousLevel", "0"
    elif tuple(words) == (1, 1, 1):
        policy, level = "AfterSpecificLevel", "1"
    else:
        return None
    return ("attrs", [("RestartPolicy", policy),
                      ("LowerLevel", level), ("UpperLevel", level)])


def delimiter_field(d):
    """The IDML field of a nested style delimiter (the string form the
    model's ``nested_delimiter`` returns)."""
    if d.startswith("char:"):
        return ("Delimiter", "string", d[5:])
    names = {
        "dropcap": "Dropcap", "any_word": "AnyWord",
        "any_character": "AnyCharacter", "tabs": "Tabs",
        "end_nested_style": "EndNestedStyle", "digits": "Digits",
        "sentence": "Sentence", "repeat": "Repeat",
    }
    return ("Delimiter", "enumeration", names[d])


def _manual_kerning(attrs, plain):
    """Sets ``KerningMethod`` in ``plain`` for a run list with manual
    kerning; see ``run_attrs``."""
    if attrs.get(0x1B07) is None:
        return
    v = attrs.get(0x1B13)
    if v is not None and v.as_f64() == KERNING_NONE:
        return
    i = None
    for j, (n, _) in enumerate(plain):
        if n == "KerningValue":
            i = j
            break
    if i is None:
        return
    method = "$ID/None" if plain[i][1] == "0" else "$ID/Manual"
    for j, (n, _) in enumerate(plain):
        if n == "KerningMethod":
            plain[j] = (n, method)
            return
    plain.insert(i, ("KerningMethod", method))


# --- value conversion (kind.rs) ----------------------------------------

def is_code(kind):
    """Whether a value without an IDML value is reported to
    ``indd audit`` as an unknown code."""
    return kind[0] in ("enum", "enum_string", "builtin", "stroke_type",
                       "bool", "restart_policies")


def _text(ty, s):
    return (ty, ("text", s))


def value(w, kind, v):
    """The IDML type and value of ``v`` written as ``kind``; ``None``
    if it has no IDML value."""
    k = kind[0]
    number = v.as_f64()
    if number is None:
        number = float(v.as_u32()) if v.as_u32() is not None else None
    if k == "number":
        return _text("unit", num(number)) if number is not None else None
    if k == "range":
        lo, hi = kind[1], kind[2]
        f = v.as_f64()
        if f is not None and lo <= f <= hi:
            return _text("unit", num(f))
        return None
    if k == "percent":
        f = v.as_f64()
        return _text("unit", num(f * 100.0)) if f is not None else None
    if k == "scale":
        f = v.as_f64()
        if f is None:
            return None
        from .idml_xml import round as _round
        return _text("unit", num(_round(f * kind[1])))
    if k == "kerning":
        f = v.as_f64()
        if f is None:
            return None
        if f == KERNING_NONE:
            return _text("unit", "1e+11")
        # Adding 0 turns -0 into 0.
        import math
        return _text("unit", num(math.floor(f * 1000.0 + 0.5) + 0.0 if f >= 0
                                 else -math.floor(-f * 1000.0 + 0.5) + 0.0))
    if k == "integer":
        u = v.as_u32()
        return _text("long", str(u)) if u is not None else None
    if k == "edge_tint":
        if number is None:
            return None
        return _text("unit", num(100.0 if number == -1.0 else number))
    if k == "tint":
        if number is None:
            return None
        return _text("unit", num(-1.0 if number == 100.0 else number))
    if k == "nonzero":
        u = v.as_u32()
        if u is not None and u != 0:
            return _text("long", str(u))
        return None
    if k == "number_or":
        f = v.as_f64()
        if f is None:
            return None
        if f == kind[1]:
            return _text("enumeration", kind[2])
        return _text(kind[3], num(f))
    if k == "leading":
        f = v.as_f64()
        if f is None:
            return None
        if f < 0.0:
            return _text("enumeration", "Auto")
        return _text("unit", num(f))
    if k == "equals":
        u = v.as_u32()
        if u is None:
            return None
        return _text("boolean", "true" if u == kind[1] else "false")
    if k == "bool":
        u = v.as_u32()
        if u is None:
            return None
        if u == 0:
            return _text("boolean", "false")
        if u == 1:
            return _text("boolean", "true")
        return None
    if k == "enum_string":
        u = v.as_u32()
        if u is None:
            return None
        for code, name in kind[1]:
            if code == u:
                return _text("string", name)
        return None
    if k == "enum":
        u = v.as_u32()
        if u is None:
            return None
        for code, name in kind[1]:
            if code == u:
                return _text("enumeration", name)
        return None
    if k == "builtin":
        if v.kind == "ref_or_code":
            ref, code = v.data
            if ref == 0:
                for c, name in kind[2]:
                    if c == code:
                        return _text("object", f"{kind[1]}{name}")
            # A custom stroke style, by its UID.
            if code == CUSTOM_STROKE_TYPE and kind[2] is STROKE_TYPES:
                for s in w.doc.stroke_styles:
                    if s.uid == ref:
                        return _text("object", custom_stroke_ref(s))
        return None
    if k == "stroke_type":
        if v.kind == "words":
            code, second = v.data
            if second != 0:
                return None
            if code == CELL_NO_STROKE_TYPE:
                return _text("object", "n")
            for c, name in STROKE_TYPES:
                if c == code:
                    return _text("object", f"StrokeStyle/$ID/{name}")
        return None
    if k == "point":
        if v.kind == "point":
            x, y = v.data
            return _text("unit", nums([x, y]))
        return None
    if k == "swatch":
        u = v.as_u32()
        if u is None:
            return None
        s = w.doc.swatches.get(u)
        return _text("object", s) if s is not None else None
    if k == "swatch_or_text":
        u = v.as_u32()
        if u is None:
            return None
        if u == 0:
            return _text("string", "Text Color")
        s = w.doc.swatches.get(u)
        return _text("object", s) if s is not None else None
    if k == "swatch_or_none":
        u = v.as_u32()
        if u is None:
            return None
        if u == 0:
            return _text("object", "Swatch/None")
        s = w.doc.swatches.get(u)
        return _text("object", s) if s is not None else None
    if k == "numbering_list":
        u = v.as_u32()
        if u is None:
            return None
        for lst in w.doc.numbering_lists:
            if lst.uid == u:
                return _text("object", f"NumberingList/{self_name(lst.name)}")
        return None
    if k == "font":
        u = v.as_u32()
        if u is None:
            return None
        f = w.doc.fonts.get(u)
        if f is None:
            return None
        c = w.composite_font_of(f)
        if c is not None:
            return _text("object", f"CompositeFont/{self_name(c.name.idml())}")
        return _text("string", w.family_names(f)[0])
    if k == "family_name":
        u = v.as_u32()
        if u is None:
            return None
        f = w.doc.fonts.get(u)
        if f is None:
            return None
        return _text("string", w.family_names(f)[0])
    if k == "font_or_none":
        u = v.as_u32()
        if u is None:
            return None
        if u == 0:
            return _text("string", "$ID/")
        f = w.doc.fonts.get(u)
        if f is None:
            return None
        return _text("string", w.family_names(f)[0])
    if k == "string":
        s = v.as_string()
        return _text("string", s) if s is not None else None
    if k == "dashes":
        if v.kind == "dashes":
            lengths, _corner = v.data
            return _text("unit", nums(lengths))
        return None
    if k == "dash_corner":
        if v.kind == "dashes" and v.data[1] == 3:
            return _text("enumeration", "DashesAndGaps")
        return None
    if k == "glyph_name":
        if v.kind == "glyph" and v.data:
            return _text("string", f"$ID/{v.data}")
        return None
    if k == "features":
        if v.kind == "features" and len(v.data) == 1:
            tag, n = v.data[0]
            return ("list", ("lists", [[("string", f"$ID/{tag}"),
                                        ("long", str(n))]]))
        return None
    if k == "string_or_nothing":
        s = v.as_string()
        if s is None:
            return None
        if s == "":
            return _text("enumeration", "Nothing")
        return _text("string", s)
    if k == "empty_as_nothing":
        s = v.as_string()
        if s is not None and s == "":
            return _text("enumeration", "Nothing")
        return None
    if k == "text":
        if v.kind == "text" and v.data:
            return _text("string", v.data)
        return None
    if k == "language":
        u = v.as_u32()
        if u is None:
            return None
        lang = w.doc.languages.get(u)
        if lang is None:
            return None
        # `Neutral` is `[No Language]` without `$ID/`.
        if lang == "Neutral":
            return _text("string", "[No Language]")
        from .object import builtin_key
        return _text("string", builtin_key(lang))
    if k == "character_style":
        u = v.as_u32()
        if u is None:
            return None
        return _text("object", w.style_ref(u if u != 0 else None, False))
    if k == "cell_style":
        u = v.as_u32()
        if u is None:
            return None
        if u == 0:
            return _text("object", "n")
        s = w.doc.cell_styles.get(u)
        if s is None:
            return None
        return _text("object", w.table_style_ref("CellStyle", s))
    if k == "cjk_set":
        u = v.as_u32()
        if u is None:
            return None
        if u == 0:
            return _text("enumeration", "Nothing")
        t = None
        for cand in w.doc.cjk_tables:
            if cand.uid == u:
                t = cand
                break
        if t is None:
            return None
        if t.name.builtin:
            e = builtin_cjk_table(t.name.name)
            if e is None:
                return None
            return _text("enumeration", e)
        if not t.mojikumi:
            tag = "KinsokuTable"
        elif t.custom_mojikumi is not None:
            tag = "MojikumiTable"
        else:
            return None
        return _text("object", f"{tag}/{self_name(t.name.name)}")
    if k == "tab_list":
        if v.kind == "tab_list":
            items = tab_list(v.data)
            if items is not None:
                return ("list", ("list", items))
        return None
    if k == "nested_styles":
        if v.kind == "style_list":
            _count, items = v.data
            if items is not None and items[0] == "nested" and items[1]:
                return ("list", ("list", nested_styles(w, items[1])))
        return None
    if k == "grep_styles":
        if v.kind == "style_list":
            _count, items = v.data
            if items is not None and items[0] == "grep" and items[1]:
                return ("list", ("list", grep_styles(w, items[1])))
        return None
    if k == "line_styles":
        if v.kind == "style_list":
            _count, items = v.data
            if items is not None and items[0] == "line" and items[1]:
                return ("list", ("list", line_styles(w, items[1])))
        return None
    if k == "bullet_char":
        if v.kind == "bullet_char":
            bkind, bvalue = v.data
            got = bullet_char(bkind, bvalue)
            if got is not None:
                return ("", got)
        return None
    if k == "restart_policies":
        if v.kind == "words3":
            got = restart_policies(v.data)
            if got is not None:
                return ("", got)
        return None
    return None


def value_text(w, kind, v):
    """The IDML text of ``v`` written as ``kind``, for a plain
    attribute."""
    got = value(w, kind, v)
    if got is None:
        return None
    ty, val = got
    if val[0] == "text":
        return val[1]
    return None


def attr_values(w, attrs, table):
    """The IDML values of the attributes of ``attrs`` listed in
    ``table`` (ID, IDML name, kind), in table order. Values without an
    IDML value are left out; unknown codes are reported to
    ``indd audit``."""
    out = []
    for attr_id, name, kind in table:
        v = attrs.get(attr_id)
        if v is None:
            continue
        text = value_text(w, kind, v)
        if text is not None:
            out.append((name, text))
        elif is_code(kind):
            attrs.unknown_code(attr_id, v)
    return out


def text_attrs(w, attrs):
    """IDML attributes and properties for a text attribute list."""
    return text_attrs_of(w, attrs, False)


def run_attrs(w, attrs):
    """Like ``text_attrs``, for the attributes of a story run: a run
    that stores both a kerning method and a manual kerning value has
    the method ``$ID/Manual``, or ``$ID/None`` for a value of 0,
    whatever method is stored."""
    plain, props = text_attrs(w, attrs)
    _manual_kerning(attrs, plain)
    return plain, props


def text_attrs_of(w, attrs, character_style):
    """Like ``text_attrs``; a character style (``character_style``)
    names a composite font by its family name."""
    plain = []
    props = []
    for attr_id, name, kind, in_props in TEXT_ATTRS:
        if kind[0] == "font" and character_style:
            kind = ("family_name",)
        # IDML has `MergeConsecutiveParaBorders` from version 13.1.
        if attr_id == 0x1DF1F and not w.saved_by((13, 1)):
            continue
        v = attrs.get(attr_id)
        if v is None:
            continue
        out = value(w, kind, v)
        if out is None and is_code(kind):
            attrs.unknown_code(attr_id, v)
        if out is not None:
            ty, val = out
            if in_props:
                props.append((name, ty, val))
            elif val[0] == "text":
                plain.append((name, val[1]))
    for ids, names, kind in TIED_TEXT_ATTRS:
        v = attrs.get(ids[0])
        if v is None:
            continue
        if any(attrs.get(attr_id) != v for attr_id in ids[1:]):
            continue
        out = value(w, kind, v)
        if out is None:
            if is_code(kind):
                attrs.unknown_code(ids[0], v)
            continue
        ty, val = out
        if val[0] == "text":
            plain.extend((n, val[1]) for n in names)
    for ids, names, known in PAIRED_TEXT_ATTRS:
        a = attrs.get(ids[0])
        b = attrs.get(ids[1])
        if a is None or b is None:
            continue
        codes = [a.as_u32(), b.as_u32()]
        for c, values in known:
            if codes == [c[0], c[1]]:
                plain.append((names[0], values[0]))
                plain.append((names[1], values[1]))
                break
        else:
            attrs.unknown_code(ids[0], a)
    # An empty font style is written as the style of the list's font
    # family named `Regular`, else of its first font.
    for j, (n, v) in enumerate(plain):
        if n == "FontStyle" and v == "":
            f = None
            ref = attrs.get(0x1B2B)
            if ref is not None:
                u = ref.as_u32()
                if u is not None:
                    f = w.doc.fonts.get(u)
            if f is not None and f.fonts:
                font = next((x for x in f.fonts if x.style == "Regular"),
                            f.fonts[0])
                plain[j] = (n, font.style)
            break
    return plain, props


def nested_styles(w, styles):
    """Records of an ``AllNestedStyles`` list."""
    out = []
    for n in styles:
        out.append([
            _applied_character_style(w, n.style),
            delimiter_field(n.delimiter),
            ("Repetition", "long", str(n.repetition)),
            ("Inclusive", "boolean", "true" if n.inclusive else "false"),
        ])
    return out


def grep_styles(w, styles):
    """Records of an ``AllGREPStyles`` list."""
    return [[_applied_character_style(w, g.style),
             ("GrepExpression", "string", g.expression)]
            for g in styles]


def line_styles(w, styles):
    """Records of an ``AllLineStyles`` list. ``RepeatLast`` is -1 in
    every sample and has no stored field."""
    return [[_applied_character_style(w, line.style),
             ("LineCount", "long", str(line.lines)),
             ("RepeatLast", "long", "-1")]
            for line in styles]


def _applied_character_style(w, uid):
    return ("AppliedCharacterStyle", "object",
            w.style_ref(uid if uid != 0 else None, False))


def item_attr_values(w, attrs):
    """Page item attributes from an attribute list, as IDML values."""
    return attr_values(w, attrs, ITEM_ATTRS)


def item_default_values(w, attrs):
    """IDML attributes of the page item defaults list."""
    out = attr_values(w, attrs, ITEM_ATTRS)
    out.extend(attr_values(w, attrs, ITEM_DEFAULT_ATTRS))
    return out


def table_attrs(w, attrs, style):
    """IDML attributes of a table or (``style``) table style attribute
    list."""
    out = attr_values(w, attrs, TABLE_ATTRS)
    out.extend(attr_values(w, attrs, GRAPHIC_CELL_ATTRS))
    out.extend(attr_values(w, attrs, TABLE_STYLE_ONLY_ATTRS if style
                           else TABLE_ONLY_ATTRS))
    for ids, names, kind, table_only in TABLE_GROUPS:
        if style and table_only:
            continue
        first = attrs.get(ids[0])
        if first is None:
            continue
        if not all(attrs.get(attr_id) == first for attr_id in ids[1:]):
            continue
        text = value_text(w, kind, first)
        if text is not None:
            out.extend((n, text) for n in names)
        elif is_code(kind):
            attrs.unknown_code(ids[0], first)
    return out


def cell_attrs(w, attrs):
    """IDML attributes of a cell attribute set other than its edges.
    Some cell attributes are written as two IDML attributes."""
    out = []
    for attr_id, names, kind in CELL_ATTRS:
        v = attrs.get(attr_id)
        if v is None:
            continue
        text = value_text(w, kind, v)
        if text is not None:
            for name in names:
                # Text cell values exist from DOM 11.
                if w.doc.version.major >= 11 or not name.startswith("Text"):
                    out.append((name, text))
        elif is_code(kind):
            attrs.unknown_code(attr_id, v)
    return out


def cell_edge_attrs(w, t, c):
    """The edge attributes of a cell. An edge value is written when
    every grid position along the edge has it, with the same value;
    the priority comes from the cell's own set."""
    edges = t.edge_formats(c)
    own = t.format(c)
    spans = [False, c.column_span > 1, False, c.row_span > 1]
    values = []
    for ids, _names, kind in CELL_EDGE_ATTRS:
        row = [None, None, None, None]
        for e in range(4):
            attr_id = ids[e]
            if ids == CELL_EDGE_PRIORITY:
                v = own.attrs.get(attr_id) if own is not None else None
                if v is None and spans[e] and any(
                        f is not None and f.attrs.get(attr_id) is not None
                        for f in edges[e]):
                    row[e] = "1"
                    continue
            else:
                vs = [f.attrs.get(attr_id) if f is not None else None
                      for f in edges[e]]
                first = vs[0] if vs else None
                v = first if all(x == first for x in vs) else None
            if v is None:
                continue
            text = value_text(w, kind, v)
            if text is not None:
                row[e] = text
            elif is_code(kind) and own is not None:
                own.attrs.unknown_code(attr_id, v)
        values.append(row)
    _edge_values(values)
    return _edge_list(values)


def _edge_list(values):
    """Edge values as IDML attributes, by edge (left, top, right,
    bottom). ``values`` follows ``CELL_EDGE_ATTRS``."""
    out = []
    for e in (0, 2, 1, 3):
        for row, (_ids, names, _kind) in zip(values, CELL_EDGE_ATTRS,
                                            strict=True):
            if row[e] is not None:
                out.append((names[e], row[e]))
                row[e] = None
    return out


def _edge_values(values):
    """Edge values of a cell written as IDML writes them: a weight is 0
    when the edge's colour is the ``None`` swatch. ``values`` follows
    ``CELL_EDGE_ATTRS`` (weights first, colours third)."""
    weights = values[0]
    colors = values[2]
    for e in range(4):
        if colors[e] == "Swatch/None" and weights[e] is not None:
            weights[e] = "0"


def cell_style_edge_attrs(w, attrs):
    """The edge attributes of a cell style: the cell edge IDs, each
    written for the rotated style edge (``CELL_STYLE_EDGE``)."""
    values = []
    for ids, _names, kind in CELL_EDGE_ATTRS:
        # Styles keep their stored tints and weights.
        if kind[0] == "edge_tint":
            kind = ("number",)
        row = [None, None, None, None]
        for e in range(4):
            v = attrs.get(ids[e])
            if v is None:
                continue
            text = value_text(w, kind, v)
            if text is not None:
                row[CELL_STYLE_EDGE[e]] = text
            elif is_code(kind):
                attrs.unknown_code(ids[e], v)
        values.append(row)
    return _edge_list(values)


def gradients(item):
    """The gradient attributes of a page item: its own, or, for a
    group, the values all its children have."""
    # A form field has the values of the items it shows, as a group
    # has those of its children.
    kind = item.kind
    shown = []
    if kind == ("group",):
        shown = list(item.children)
    elif kind[0] == "form" and FormKind.has_states(kind[1].kind):
        shown = [it for s in kind[1].states for it in s.items]
    elif kind[0] == "form":
        shown = list(item.children)
    if kind == ("group",) or kind[0] == "form":
        kids = [gradients(k) for k in shown]
        out = []
        for _attr_id, name, _default in GRADIENT_ATTRS:
            values = [next((v for n, v in kid if n == name), None)
                      for kid in kids]
            if not values:
                continue
            first = values[0]
            if first is None:
                continue
            if all(v == first for v in values[1:]):
                out.append((name, first))
        return out
    return gradient_values(item.attrs)


def gradient_values(attrs):
    """The gradient attributes of an attribute list, with the defaults
    for those it lacks."""
    out = []
    for attr_id, name, default in GRADIENT_ATTRS:
        v = attrs.get(attr_id)
        if v is not None:
            if v.kind == "point":
                text = nums(list(v.data))
            else:
                f = v.as_f64()
                text = num(f) if f is not None else None
        else:
            text = None
        out.append((name, text if text is not None else default))
    return out
