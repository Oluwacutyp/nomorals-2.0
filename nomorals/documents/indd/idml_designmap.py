"""``designmap.xml``: the document element and the document-level
elements written there (languages, colour groups, TOC styles,
cross-reference formats, hyperlinks, bookmarks, text variables, index
options, sections).

A faithful port of ``src/idml/designmap.rs`` from storytold/indd-utils
(commit 136f973, dual MIT/Apache-2.0 — see attribution.py). Helpers
that upstream defines in other writer modules live there:
``label``/``grid_data`` in idml_spread, ``motion_preset`` in
idml_animation, ``style_mappings`` in idml_storylink.
"""

from __future__ import annotations

from .idml import (
    Edition,
    alternate_layout_name,
    alternate_layouts,
    document_pages,
    number_style,
    section_ranges,
)
from .idml_animation import motion_preset
from .idml_spread import grid_data, label
from .idml_storylink import style_mappings
from .idml_values import Node, element, keyed, present, when_written
from .idml_xml import Xml, num, self_name, ui_color_property, uref, variable_name, write_properties
from .object import builtin_key


def index_header_type(group, variant):
    """IDML ``HeaderType`` of an index group by the name of its header
    variant; a variant named like its group has none."""
    if variant == "IDX_Basic":
        return "BasicLatin"
    if variant == "IDX_Spanish":
        return "Spanish"
    if variant == "IDX_Czech":
        return "Czech"
    if variant == "IDX_Russian":
        return "Russian"
    if variant == "IDX_AllHira":
        return "HiraganaAll"
    if variant == "IDX_ChinesePinyin":
        return "ChinesePinyin"
    if variant == "IDX_ChineseStroke":
        return "ChineseStrokeCount"
    if variant == "IDX_KoreanConsonant":
        return "KoreanConsonant"
    if variant == group:
        return "Nothing"
    return None


#: Languages whose hyphenation vendor is never stored in the corpus:
#: IDML writes ``$ID/`` for their empty vendor.
NO_HYPHENATION_VENDOR = [
    "Albanian", "Arabic", "Byelorussian", "Chinese: Hong Kong",
    "Chinese: Simplified", "Chinese: Taiwan", "Chinese: Traditional",
    "Icelandic", "Japanese", "Korean", "Neutral", "Vietnamese",
    "cy_GB", "en_US+Medical", "eu_ES", "fo_FO", "fy_NL", "ga_IE",
    "gl_ES", "gv_GB", "ko_KR", "la_VA", "lb_LU", "mk_MK", "mn_MN",
    "ms_MY", "ne_NE", "oc_FR", "sr_RS", "sr_RS-Cyr", "sr_RS-Lat",
    "tk_TM", "uz_UZ",
]

#: Languages whose spelling vendor is never stored in the corpus.
NO_SPELLING_VENDOR = [
    "Albanian", "Byelorussian", "Chinese: Hong Kong",
    "Chinese: Simplified", "Chinese: Taiwan", "Chinese: Traditional",
    "Icelandic", "Japanese", "Korean", "Neutral", "cy_GB", "eu_ES",
    "fo_FO", "fy_NL", "ga_IE", "gl_ES", "gv_GB", "la_VA", "lb_LU",
    "mk_MK", "mn_MN", "ms_MY", "ne_NE", "oc_FR", "sr_RS", "sr_RS-Cyr",
    "sr_RS-Lat", "tk_TM", "uz_UZ",
]

#: Quotes of a language without chunk 0x2D26 that the value files
#: lack. Every Romanian language without the chunk has these, but
#: those with it differ. The others are defaults.
QUOTES_WITHOUT_CHUNK = [
    ("Romanian", "\u201a\u2019", "\u201e\u201d"),
    ("Korean", "''", '""'),
    ("Japanese", "''", '""'),
    ("Hebrew", "''", '""'),
]


def exporter_values(w, tag):
    """Values of designmap settings that follow the InDesign that
    saved the document last, as the save history gives it. They apply
    where the INDD gives none."""
    doc = w.doc
    header = (doc.version.major, doc.version.minor)
    last = doc.last_session_version if doc.last_session_version is not None else header
    if tag == "PublishExportPreference":
        out = []
        if last == (11, 0):
            out.append(("ImageExportResolution", "Ppi72"))
        elif last >= (11, 1):
            out.append(("ImageExportResolution", "Ppi96"))
        if last >= (11, 2):
            out.append(("PublishPdf", "false"))
        return out
    if tag == "AdjustLayoutPreference":
        early_14 = last == (14, 0) and doc.last_session_patch in (0, 1)
        return [
            ("EnableAdjustLayout", "true" if early_14 else "false"),
            ("AllowFontSizeAndLeadingAdjustment", "false"),
            ("EnableAutoAdjustMargins", "false"),
        ]
    if tag == "WatermarkPreference":
        # Without chunk 0x16344 the font follows the language of the
        # assignment name.
        font = None
        if doc.assignment_name == "未指定的 InCopy 內容":
            font = ("Adobe Ming Std", "L")
        else:
            edition = w.edition()
            if edition == Edition.ENGLISH:
                font = ("Minion Pro", "Regular")
            elif edition == Edition.KOREAN:
                font = ("Adobe Myungjo Std", "M")
            elif edition == Edition.JAPANESE:
                font = ("Kozuka Mincho Pro", "R")
        if font is None:
            return []
        return [("WatermarkFontFamily", font[0]),
                ("WatermarkFontStyle", font[1])]
    return []


def designmap(w, name):
    doc = w.doc
    x = Xml()
    x.pi(f'aid style="50" type="document" readerVersion="6.0" '
         f'featureSet="257" product="{w.dom}" ')
    # The stories, then the backing story.
    stories = [uref(s.uid) for s in doc.stories]
    if doc.xml.story is not None:
        stories.append(uref(doc.xml.story.uid))
    x.start("Document") \
        .attr("xmlns:idPkg", "http://ns.adobe.com/AdobeInDesign/idml/1.0/packaging") \
        .attr("DOMVersion", w.dom) \
        .attr("Self", "d") \
        .attr("StoryList", " ".join(stories))
    # `Name` from DOM 13.
    if doc.version.major >= 13:
        x.attr("Name", name)
    x.attr("ZeroPoint", "0 0")
    if doc.active_layer is not None:
        x.attr("ActiveLayer", uref(doc.active_layer))
    # Colour settings (`model::prefs`).
    for v in doc.prefs.values:
        if v.element == "Document":
            x.attr(v.name, v.value)
    # `false` in every corpus IDML.
    x.attr("AccurateLABSpots", "false")
    # The MathML settings every IDML of DOM 20 has, and of 20.4.
    mathml = w.dom_version() >= (20, 0)
    if mathml:
        x.attr("AppliedMathMLFontSize", "10") \
            .attr("AppliedMathMLRgbColor", "0 0 0") \
            .attr("TintValue", "100")
    if w.dom_version() >= (20, 4):
        x.attr("PreferMathMLInEpubExport", "false")
    if doc.label or mathml:
        x.start("Properties")
        label(x, doc.label)
        if mathml:
            x.start("AppliedMathMLSwatch") \
                .attr("type", "enumeration") \
                .text("Nothing") \
                .end()
        x.end()
    languages(w, x)
    x.empty("idPkg:Graphic", [("src", "Resources/Graphic.xml")])
    x.empty("idPkg:Fonts", [("src", "Resources/Fonts.xml")])
    # Kinsoku tables, then mojikumi tables, as the schema orders them.
    for mojikumi in (False, True):
        for t in doc.cjk_tables:
            if t.mojikumi != mojikumi:
                continue
            tag = "MojikumiTable" if mojikumi else "KinsokuTable"
            tname = t.name.idml()
            x.start(tag) \
                .attr("Self", f"{tag}/{self_name(tname)}") \
                .attr("Name", tname)
            if t.chars is not None:
                begin, end, _unused, hanging, together = t.chars
                x.attr("CantBeginLineChars", begin) \
                    .attr("CantEndLineChars", end) \
                    .attr("HangingPunctuationChars", hanging) \
                    .attr("CantBeSeparatedChars", together)
            if t.custom_mojikumi is not None:
                custom_mojikumi(x, t.custom_mojikumi)
            x.end()
    x.empty("idPkg:Styles", [("src", "Resources/Styles.xml")])
    # The numbering lists, the default first. Without them, the
    # default list every corpus IDML has, which the root paragraph
    # style refers to.
    if doc.numbering_lists:
        lists = doc.numbering_lists
    else:
        from .model import NumberingList
        lists = [NumberingList(uid=0, name="$ID/[Default]",
                               across_stories=False, across_documents=False)]
    for lst in lists:
        x.empty("NumberingList", [
            ("Self", f"NumberingList/{self_name(lst.name)}"),
            ("Name", lst.name),
            ("ContinueNumbersAcrossStories", "true" if lst.across_stories else "false"),
            ("ContinueNumbersAcrossDocuments", "true" if lst.across_documents else "false"),
        ])
    major = doc.version.major

    def singleton(tag):
        # Every IDML from 7.5 has the tagged PDF settings, none of 7.0.
        # Every IDML from DOM 10.1 has the fixed-layout HTML export
        # settings, none of 10.0, and from DOM 21.1 the HTML5 export
        # settings, with the values all of them have.
        n = present(f"Document/{tag}", major)
        if n is None:
            since = False
            if tag == "TaggedPDFPreference":
                since = (major, doc.version.minor) >= (7, 5)
            elif tag == "HTMLFXLExportPreference":
                since = w.dom_version() >= (10, 1)
            elif tag == "Html5ExportPreference":
                since = w.dom_version() >= (21, 1)
            if since:
                n = element(f"Document/{tag}", major)
                if n is None:
                    attrs = []
                    if tag == "Html5ExportPreference":
                        attrs = [("TextExportFormat", "HtmlTag"),
                                 ("CopyFonts", "false")]
                    n = Node(tag=tag, attrs=attrs)
        if n is None:
            return
        # Values read from the INDD (`model::prefs`) first.
        ours = endnote_option(w) if tag == "EndnoteOption" else None
        if ours is None:
            ours = Node(tag=n.tag)
        for v in doc.prefs.values:
            if v.element == tag:
                ours.attrs.append((v.name, v.value))
        for k, v in exporter_values(w, tag):
            if not any(a == k for a, _ in ours.attrs):
                ours.attrs.append((k, v))
        colors = []
        for element_name, cname, rgb in doc.prefs.colors:
            if element_name == tag:
                node = ui_color_property(cname, rgb)
                if node is not None:
                    colors.append(node)
        if colors:
            ours.children.append(Node(tag="Properties", children=colors))
        ours.merge(n)
        ours.write(x)

    # A named grid without settings of its own has those of the
    # document's layout grid.
    layout_grid = None
    for tag, g, _count in doc.prefs.grids:
        if tag == "LayoutGridDataInformation":
            layout_grid = g
            break
    for ng in doc.named_grids:
        x.start("NamedGrid") \
            .attr("Self", named_grid_self(ng)) \
            .attr("Name", named_grid_name(ng))
        grid = ng.grid if ng.grid is not None else layout_grid
        if grid is not None:
            grid_data(w, x, grid)
        x.end()
    for p in doc.motion_presets:
        motion_preset(x, p)
    singleton("ConditionalTextPreference")
    x.empty("idPkg:Preferences", [("src", "Resources/Preferences.xml")])
    for tag in ["EndnoteOption", "TextFrameFootnoteOptionsObject",
                "LinkedStoryOption", "LinkedPageItemOption",
                "WatermarkPreference", "TaggedPDFPreference",
                "AdjustLayoutPreference", "HTMLFXLExportPreference",
                "PublishExportPreference", "Html5ExportPreference"]:
        singleton(tag)
    text_variables(w, x)
    x.empty("idPkg:Tags", [("src", "XML/Tags.xml")])
    for layer in doc.layers:
        if layer.internal:
            continue
        x.start("Layer") \
            .attr("Self", uref(layer.uid)) \
            .attr("Name", layer.name) \
            .attr("Visible", "true" if layer.visible else "false") \
            .attr("Locked", "true" if layer.locked else "false")
        color = None
        if layer.settings is not None:
            st = layer.settings
            x.attr("IgnoreWrap", "true" if st.ignore_wrap else "false") \
                .attr("LockGuides", "true" if st.lock_guides else "false") \
                .attr("UI", "true" if st.ui else "false") \
                .attr("Printable", "true" if st.printable else "false")
            if st.color is not None:
                color = ui_color_property("LayerColor", st.color)
        x.attrs_missing(w.observed("Layer"))
        if color is not None:
            write_properties(x, [], [color])
        x.end()
    for s in doc.master_spreads:
        x.empty("idPkg:MasterSpread", [
            ("src", f"MasterSpreads/MasterSpread_u{s.uid:x}.xml")])
    for s in doc.spreads:
        x.empty("idPkg:Spread", [
            ("src", f"Spreads/Spread_u{s.uid:x}.xml")])
    pages = document_pages(doc)
    layout_lengths, _ = alternate_layouts(doc)
    for k, (section, first, length) in enumerate(section_ranges(doc)):
        x.start("Section") \
            .attr("Self", uref(section.uid)) \
            .attr("Length", str(length)) \
            .attr("Name", section.prefix) \
            .attr("ContinueNumbering", "true" if section.continue_numbering else "false") \
            .attr("IncludeSectionPrefix", "true" if section.include_prefix else "false") \
            .attr("Marker", section.marker) \
            .attr("PageStart", uref(pages[first]))
        if doc.version.major >= 8:
            alt_name = alternate_layout_name(section)
            if alt_name is not None:
                x.attr("AlternateLayoutLength", str(layout_lengths[k])) \
                    .attr("AlternateLayout", alt_name)
        # The pagination values of DOM 8.
        sec = element("Section", doc.version.major)
        if sec is not None:
            x.attrs_missing(sec.attrs)
        # IDML gives a start number only to sections that restart
        # numbering.
        if not section.continue_numbering:
            x.attr("PageNumberStart", str(section.start))
        x.attr("SectionPrefix", section.prefix)
        style = number_style(section.style)
        if style is not None:
            x.start("Properties") \
                .start("PageNumberStyle") \
                .attr("type", "enumeration") \
                .text(style) \
                .end() \
                .end()
        x.end()
    # Document users. IDML names `$ID/Unknown User Name` a user whose
    # name is the saving edition's name for an unknown user; without
    # the edition, a user with flag 2. The colours are left out.
    unknown = w.saving_edition()
    unknown_names = Edition.unknown_user(unknown) if unknown is not None else None
    for i, (flag, uname) in enumerate(doc.users):
        placeholder = uname in unknown_names \
            if unknown_names is not None else flag == 2
        disp = "$ID/Unknown User Name" if placeholder else uname
        x.empty("DocumentUser", [
            ("Self", f"dDocumentUser{i:x}"),
            ("UserName", disp),
        ])
    cross_reference_formats(w, x)
    if doc.index is not None:
        index_xml(x, doc.index)
    x.empty("idPkg:BackingStory", [("src", "XML/BackingStory.xml")])
    for s in doc.stories:
        x.empty("idPkg:Story", [
            ("src", f"Stories/Story_u{s.uid:x}.xml")])
    style_mappings(w, x, "d", doc.style_mappings)
    hyperlinks(w, x)
    color_groups(w, x)
    for i, b in enumerate(doc.bullets):
        kinds = ["UnicodeOnly", "UnicodeWithFont", "GlyphWithFont"]
        if b.kind >= len(kinds):
            continue
        if b.font == 0:
            font = "$ID/"
        else:
            fam = doc.fonts.get(b.font)
            if fam is None:
                continue
            font = w.family_names(fam)[0]
        x.start("ABullet") \
            .attr("Self", f"dABullet{i:x}") \
            .attr("CharacterType", kinds[b.kind]) \
            .attr("CharacterValue", str(b.value))
        write_properties(x, [
            ("BulletsFont", "string", ("text", font)),
            ("BulletsFontStyle", "string", ("text", b.font_style.idml())),
        ])
        x.end()
    # Index sort options: the groups in their order, with the header
    # type each group and header variant has in IDML.
    for i, (iname, include, variant) in enumerate(doc.index_groups):
        x.start("IndexingSortOption") \
            .attr("Self", f"dIndexingSortOptionn{iname}") \
            .attr("Name", builtin_key(iname)) \
            .attr("Include", "true" if include else "false") \
            .attr("Priority", str(i))
        if variant is not None:
            h = index_header_type(iname, variant)
            if h is not None:
                x.attr("HeaderType", h)
        x.end()
    # InCopy assignments: the values every IDML has on them; the name
    # depends on the computer that exported the IDML.
    assignment = element("Assignment", doc.version.major)
    if assignment is not None:
        for uid in doc.assignments:
            a = Node(tag=assignment.tag, attrs=list(assignment.attrs),
                     text=assignment.text,
                     children=list(assignment.children))
            a.attrs.insert(0, ("Self", uref(uid)))
            a.write(x)
    articles(w, x)
    x.end()
    return x.finish()




def articles(w, x):
    """The articles, with the page items each lists. An unwritten
    member is left out, with a warning."""
    if not w.doc.articles:
        return
    written = set(w.written_page_items().keys())
    for a in w.doc.articles:
        aid = uref(a.uid)
        x.start("Article") \
            .attr("Self", aid) \
            .attr("Name", a.name) \
            .attr("ArticleExportStatus", "true")
        i = 0
        for m in a.members:
            if m not in written:
                w.warn(f"article {a.uid}: member {m} left out: not written")
                continue
            x.start("ArticleMember") \
                .attr("Self", f"{aid}ArticleMember{i}") \
                .attr("ItemRef", uref(m))
            x.end()
            i += 1
        x.end()


def named_grid_name(ng):
    """The IDML name of a named grid: the built-in key where it has
    one."""
    return builtin_key(ng.name) if ng.builtin else ng.name


def named_grid_self(ng):
    return f"NamedGrid/{self_name(named_grid_name(ng))}"


def named_grid_ref(w, applied):
    """``AppliedNamedGrid`` of a story or object style: ``applied`` is
    ``None`` for no value, ``("none",)`` for none, or ``("uid", uid)``
    for a grid UID. Returns ``"n"`` for none, the grid's ``Self``
    for a grid of the document, ``None`` otherwise."""
    if applied is None:
        return None
    if applied[0] == "none":
        return "n"
    for g in w.doc.named_grids:
        if g.uid == applied[1]:
            return named_grid_self(g)
    return None


def toc_style_ref(t):
    """The IDML reference of a TOC style."""
    name = builtin_key(t.name) if t.builtin else t.name
    return f"TOCStyle/{self_name(name)}"


def toc_styles(w, x):
    """The TOC styles, each with its entry styles. Called from the
    Styles part, not from designmap."""
    for t in w.doc.toc_styles:
        name = builtin_key(t.name) if t.builtin else t.name
        title_style = t.title_style if t.title_style != 0 else None
        x.start("TOCStyle") \
            .attr("Self", toc_style_ref(t)) \
            .attr("TitleStyle", w.style_ref(title_style, True)) \
            .attr("Title", t.title) \
            .attr("Name", name) \
            .attr("IncludeBookDocuments", str(t.include_book_documents).lower()) \
            .attr("CreateBookmarks", str(t.create_bookmarks).lower())
        major = w.doc.version.major

        def flag(i, t=t):
            return t.flags[i] if i < len(t.flags) else None

        if flag(0) == 0:
            x.attr("SetStoryDirection", "Horizontal")
        elif flag(0) == 1:
            x.attr("SetStoryDirection", "Vertical")
        if flag(1) == 0:
            x.attr("NumberedParagraphs", "IncludeFullParagraph")
        elif flag(1) == 2:
            x.attr("NumberedParagraphs", "ExcludeNumbers")
        if major >= 9 and flag(2) is not None:
            x.attr("MakeAnchor", str(flag(2) != 0).lower())
        if major >= 13 and flag(3) is not None:
            x.attr("RemoveForcedLineBreak", str(flag(3) != 0).lower())
        x.attrs_missing(w.observed("TOCStyle"))
        x.attrs_missing(when_written("TOCStyle", major))
        observed = w.observed("TOCStyleEntry")
        for i, e in enumerate(t.entries):
            toc_entry(w, x, t.uid, i, e, observed)
        x.end()


def toc_entry(w, x, toc, i, e, observed):
    """An entry of a table of contents style."""
    name = e.name.replace("\uE00B", ":")
    if e.builtin:
        name = f"$ID/{name}"
    position = {0: "AfterEntry", 1: "BeforeEntry", 2: "None"}.get(
        e.page_number_position)
    x.start("TOCStyleEntry") \
        .attr("Self", f"u{toc:x}TOCStyleEntry{i:x}") \
        .attr("Name", name) \
        .attr("Level", str(e.level))
    if position is not None:
        x.attr("PageNumberPosition", position)
    x.attr("Separator", toc_separator(e.separator))
    x.attrs_missing(observed)

    def style(uid, paragraph, same):
        if uid == 0:
            return ("string", same)
        if uid not in w.doc.styles:
            return None
        r = w.style_ref(uid, paragraph)
        for root in ("$ID/[No paragraph style]", "$ID/[No character style]"):
            if r.endswith(root):
                return ("string", root)
        return ("object", r)

    props = []
    for key, value in [
            ("FormatStyle", style(e.format_style, True, "$ID/kSameStyleWithBracket")),
            ("PageNumberStyle", style(e.page_number_style, False, "$ID/[Same Style]")),
            ("SeparatorStyle", style(e.separator_style, False, "$ID/[Same Style]")),
    ]:
        if value is not None:
            props.append((key, value))
    if props:
        x.start("Properties")
        for k, (ty, v) in props:
            x.start(k).attr("type", ty).text(v).end()
        x.end()
    x.end()


def languages(w, x):
    """The document's languages, in UID order."""
    for lang in sorted(w.doc.language_list,
                       key=lambda language: language.uid):
        neutral = lang.name == "Neutral"

        def lname(s, neutral=neutral):
            return "$ID/[No Language]" if neutral else builtin_key(s)

        full = lname(lang.name)
        x.start("Language") \
            .attr("Self", f"Language/{self_name(full)}") \
            .attr("Name", full)
        # The quotes stored with the language, else those every IDML
        # has for its name, else the defaults.
        if lang.quotes is not None:
            single, double = lang.quotes
            x.attr("SingleQuotes", single).attr("DoubleQuotes", double)
        else:
            q = keyed("Language", full)
            if q is not None:
                for k in ("SingleQuotes", "DoubleQuotes"):
                    v = q.attr(k)
                    if v is not None:
                        x.attr(k, v)
            else:
                for lang_name, s, d in QUOTES_WITHOUT_CHUNK:
                    if lang.name == lang_name:
                        x.attr("SingleQuotes", s).attr("DoubleQuotes", d)
                        break
        x.attr("PrimaryLanguageName", lname(lang.primary)) \
            .attr("SublanguageName", lname(lang.sub)) \
            .attr("Id", str(lang.id))
        if lang.vendors is not None:
            (sf, spelling), (hf, hyphenation) = lang.vendors

            def vendor(flag, text, without, lang=lang):
                if flag == 1:
                    return text
                if flag == 0 and text == "" and lang.name in without:
                    return "$ID/"
                if flag == 3:
                    return "$ID/"
                if flag == 10:
                    return builtin_key(text)
                return None

            v = vendor(hf, hyphenation, NO_HYPHENATION_VENDOR)
            if v is not None:
                x.attr("HyphenationVendor", v)
            v = vendor(sf, spelling, NO_SPELLING_VENDOR)
            if v is not None:
                x.attr("SpellingVendor", v)
        x.end()


def color_groups(w, x):
    """Colour groups and their swatches."""
    for i, g in enumerate(w.doc.color_groups):
        # Named by UID before DOM 11.3.
        gid = uref(g.uid) if w.dom_version() < (11, 3) else \
            f"ColorGroup/{self_name(g.name)}"
        x.start("ColorGroup") \
            .attr("Self", gid) \
            .attr("Name", g.name) \
            .attr("IsRootColorGroup", str(i == 0).lower())
        for n, s in enumerate(g.swatches):
            ref = w.doc.swatches.get(s)
            if ref is not None:
                x.empty("ColorGroupSwatch", [
                    ("Self", f"{uref(g.uid)}ColorGroupSwatch{n:x}"),
                    ("SwatchItemRef", ref),
                ])
        x.end()


def cross_reference_formats(w, x):
    """Cross-reference formats."""
    for f in w.doc.cross_reference_formats.values():
        fid = uref(f.uid)
        x.start("CrossReferenceFormat") \
            .attr("Self", fid) \
            .attr("Name", f.name)
        if f.character_style == 0:
            x.attr("AppliedCharacterStyle", "n")
        for i, b in enumerate(f.blocks):
            kind = b.type_name()
            if kind is None:
                continue
            x.start("BuildingBlock") \
                .attr("Self", f"{fid}BuildingBlock{i}") \
                .attr("BlockType", kind)
            if b.zero_fields:
                x.attr("AppliedCharacterStyle", "n")
            x.attr("CustomText", b.text.idml())
            if b.zero_fields:
                x.attr("AppliedDelimiter", "$ID/") \
                    .attr("IncludeDelimiter", "false")
            x.end()
        x.end()


def endnote_option(w):
    """``EndnoteOption`` from preferences chunk 0x2261E; without the
    chunk the defaults, whose title and marker position follow the
    saving edition and the language of the last session."""
    n = Node(tag="EndnoteOption")
    e = w.doc.prefs.endnotes
    if e is None:
        n.attrs.append(("EndnoteMarkerStyle", w.style_ref(None, False)))
        n.attrs.append(("EndnoteTextStyle", w.style_ref(None, True)))
        saved = w.saving_edition()
        title = None
        if saved is not None:
            titles = {
                "english": "Endnotes",
                "korean": "미주",
                "german": "Endnoten",
                "french": "Notes de fin",
                "italian": "Note di chiusura",
                "dutch": "Eindnoten",
            }
            if saved == "japanese":
                title = "文末脚注" if w.dom_version()[0] <= 13 else \
                    "後注" if w.dom_version()[0] >= 19 else None
            else:
                title = titles.get(saved)
        if title is not None:
            n.attrs.insert(0, ("EndnoteTitle", title))
        japanese = w.japanese_session()
        if japanese is not None:
            if japanese:
                separator, position = "\u3000", "RubyMarker"
            else:
                separator, position = "\t", "SuperscriptMarker"
            if saved is not None and saved != "other":
                position = "RubyMarker" if saved == "japanese" else \
                    "SuperscriptMarker"
            n.attrs.append(("EndnoteSeparatorText", separator))
            n.children.append(Node(tag="Properties", children=[
                Node(tag="EndnoteMarkerPositioning",
                     attrs=[("type", "enumeration")], text=position)]))
        return n
    n.attrs.append(("EndnoteTitle", e.title))
    n.attrs.append(("EndnoteTitleStyle", w.style_ref(e.title_style, True)))
    n.attrs.append(("StartEndnoteNumberAt", str(e.start_at)))
    n.attrs.append(("EndnoteMarkerStyle", w.style_ref(e.marker_style, False)))
    n.attrs.append(("EndnoteTextStyle", w.style_ref(e.text_style, True)))
    n.attrs.append(("EndnoteSeparatorText", e.separator))
    props = []
    if e.numbering == 0xCA07:
        props.append(("EndnoteNumberingStyle", "Arabic"))
    if e.restart == 0:
        props.append(("RestartEndnoteNumbering", "Continuous"))
    if e.positioning == 1:
        props.append(("EndnoteMarkerPositioning", "SuperscriptMarker"))
    elif e.positioning == 3:
        props.append(("EndnoteMarkerPositioning", "RubyMarker"))
    n.children.append(Node(tag="Properties", children=[
        Node(tag=name, attrs=[("type", "enumeration")], text=value)
        for name, value in props]))
    return n


def written_sources(w):
    """UIDs of the hyperlink text sources written in the stories."""
    return {r.source for s in w.doc.stories for r in s.sources
            if r.source in w.doc.text_sources}


def written_text_destinations(w):
    """UIDs of the text and paragraph destinations written in
    stories."""
    return {d.uid for s in w.doc.stories
            for dests in s.text_destinations.values() for d in dests}


def written_items(w):
    """UIDs of the page items written."""
    return set(w.written_page_items().keys())


def hyperlinks(w, x):
    """Destinations, hyperlinks and bookmarks, in schema order."""
    doc = w.doc
    # Text and paragraph destinations are written in the stories, at
    # their position; only those found there can be referred to.
    in_text = written_text_destinations(w)
    dests = [d for d in doc.destinations
             if d.kind[0] not in ("text", "paragraph")
             or d.uid in in_text]
    # Schema order: page, URL, external page destinations; each
    # sorted by key.
    rank = {"page": 0, "url": 1}.get

    def dest_key(d):
        return (rank(d.kind[0]) if d.kind[0] in ("page", "url") else 2,
                d.key is not None, d.key if d.key is not None else 0)

    dests.sort(key=dest_key)

    def view_setting(view, zoom):
        if view == 0:
            x.attr("ViewSetting", "Fixed")
        elif view == 1:
            x.attr("ViewSetting", "FitWindow")
        if zoom is not None:
            x.attr("ViewPercentage", num(zoom * 100.0))

    for d in dests:
        bounds = None
        if d.kind[0] == "page":
            _, page, zoom, view, b = d.kind
            # NameManually is true on every corpus page destination.
            x.start("HyperlinkPageDestination") \
                .attr("Self", d.reference()) \
                .attr("Name", d.name) \
                .attr("NameManually", "true") \
                .attr("DestinationPage", uref(page))
            view_setting(view, zoom)
            bounds = b
        elif d.kind[0] == "url":
            _, url = d.kind
            x.start("HyperlinkURLDestination") \
                .attr("Self", d.reference()) \
                .attr("Name", d.name) \
                .attr("DestinationURL", url)
        elif d.kind[0] == "external_page":
            _, page_index, zoom, view, b, link, file_name = d.kind
            x.start("HyperlinkExternalPageDestination") \
                .attr("Self", d.reference())
            # The name is not stored; IDML makes it from the file
            # name and the page (shown for the Fixed view only).
            if view == 0 and file_name is not None:
                x.attr("Name", f"{file_name} - Page {page_index + 1} [Fixed]")
            x.attr("DestinationPageIndex", str(page_index + 1))
            view_setting(view, zoom)
            bounds = b
        else:
            continue
        x.attr("Hidden", str(d.hidden).lower())
        if d.key is not None:
            x.attr("DestinationUniqueKey", str(d.key))
        if bounds is not None:
            left, top, right, bottom = bounds

            def bound(v):
                return "1e+256" if v == 1e256 else num(v)

            x.start("Properties") \
                .start("ViewBounds") \
                .attr("Top", bound(top)) \
                .attr("Left", bound(left)) \
                .attr("Bottom", bound(bottom)) \
                .attr("Right", bound(right)) \
                .end() \
                .end()
        x.end()
    items = written_items(w)
    sources = written_sources(w)
    for s in doc.page_item_sources:
        if s.item not in items:
            continue
        x.start("HyperlinkPageItemSource") \
            .attr("Self", uref(s.uid)) \
            .attr("Name", s.name) \
            .attr("SourcePageItem", uref(s.item)) \
            .attr("Hidden", str(s.hidden).lower()) \
            .end()
        sources.add(s.uid)
    for h in sorted(doc.hyperlinks, key=lambda h: h.uid):
        # A hyperlink into another document has a destination that is
        # not converted: IDML writes it as a list, which is left out.
        # Otherwise the destination with the same key; `n` for a
        # hyperlink without one.
        dest = None
        if not h.other_document:
            for d in dests:
                if h.destination is not None and d.uid == h.destination:
                    dest = d.reference()
                    break
                if h.destination is None and h.key is not None \
                        and d.key == h.key:
                    dest = d.reference()
                    break
            if dest is None and h.kind_is_none():
                dest = "n"
        if dest is None and not h.other_document and not h.kind_is_none():
            continue
        if h.source == 0 or h.source not in sources:
            continue
        x.start("Hyperlink") \
            .attr("Self", uref(h.uid)) \
            .attr("Name", h.name) \
            .attr("Source", uref(h.source))
        if h.appearance_known:
            x.attr("Visible", "false")
            if h.highlight == 0:
                x.attr("Highlight", "None")
            elif h.highlight == 1:
                x.attr("Highlight", "Invert")
            x.attr("Width", "Thin").attr("BorderStyle", "Solid")
        x.attr("Hidden", str(h.hidden).lower())
        if h.key is not None:
            x.attr("DestinationUniqueKey", str(h.key))
        x.start("Properties")
        # Black on every corpus hyperlink with the known appearance,
        # apart from those to a page of another document.
        if h.appearance_known and not h.to_external_page():
            x.start("BorderColor") \
                .attr("type", "enumeration") \
                .text("Black") \
                .end()
        if dest is not None:
            x.start("Destination").attr("type", "object").text(dest).end()
        x.end().end()
    top = [doc.bookmarks[u] for u in doc.bookmark_order
           if u in doc.bookmarks
           and doc.bookmarks[u].parent not in doc.bookmarks]
    for b in top:
        bookmark(w, x, b, 0, in_text)


def bookmark(w, x, b, depth, in_text):
    doc = w.doc
    dest = next((d for d in doc.destinations if d.uid == b.destination), None)
    if dest is None:
        return
    if dest.kind[0] in ("text", "paragraph") and dest.uid not in in_text:
        return
    x.start("Bookmark") \
        .attr("Self", uref(b.uid)) \
        .attr("Name", b.name) \
        .attr("Destination", dest.reference())
    if depth < 64:
        for c in b.children:
            child = doc.bookmarks.get(c)
            if child is not None:
                bookmark(w, x, child, depth + 1, in_text)
    x.end()


def text_variables(w, x):
    """The text variables, sorted by name."""
    for v in sorted(w.doc.text_variables, key=lambda v: v.name):
        name = variable_name(v.name)
        x.start("TextVariable") \
            .attr("Self", f"dTextVariablen{name}") \
            .attr("Name", name)
        t = v.type_name()
        if t is not None:
            x.attr("VariableType", t)
        variable_preference(w, x, v)
        x.end()


def variable_preference(w, x, v):
    """The settings element of a text variable. Attributes other than
    the date format and the running header's style are not located in
    the INDD; they are written only for variables whose unidentified
    fields hold the values of every corpus sample, and then with the
    values those samples have in IDML."""
    same = v.as_in_samples

    def text(name):
        if v.settings is not None:
            t = v.settings.before if name == "TextBefore" else v.settings.after
            x.attr(name, t.replace("\u3000", "^("))
        elif same:
            x.attr(name, "")

    if v.kind in (0xCAA1, 0xCAAB, 0xCAAC):
        x.start("DateVariablePreference")
        text("TextBefore")
        x.attr("Format", v.text)
        text("TextAfter")
        x.end()
    elif v.kind == 0xCAAA:
        x.start("MatchParagraphStylePreference")
        text("TextBefore")
        text("TextAfter")
        x.attr("AppliedParagraphStyle", w.style_ref(v.style, True))
        if v.settings is not None:
            if v.settings.c == 0:
                x.attr("SearchStrategy", "FirstOnPage")
            elif v.settings.c == 1:
                x.attr("SearchStrategy", "LastOnPage")
            if v.settings.a in (0, 0xCAC9):
                x.attr("ChangeCase", "None")
            elif v.settings.a == 0xCAD0:
                x.attr("ChangeCase", "Sentencecase")
        # Every corpus sample is `false`; no field is known for it.
        if same or (v.settings is not None and v.settings.c <= 1):
            x.attr("DeleteEndPunctuation", "false")
        x.end()
    elif v.kind == 0xCAA9 and v.settings is not None:
        # Chapter numbers: the format code is the first u32.
        fmt = {0: "Current", 0x4C15: "Arabic"}.get(v.settings.a)
        x.start("ChapterNumberVariablePreference")
        text("TextBefore")
        if fmt is not None:
            x.attr("Format", fmt)
        text("TextAfter")
        x.end()
    elif not same:
        pass
    elif v.kind == 0xCAA3:
        x.start("CustomTextVariablePreference") \
            .start("Properties") \
            .start("Contents") \
            .attr("type", "string") \
            .end() \
            .end() \
            .end()
    elif v.kind == 0xCAA6:
        x.empty("FileNameVariablePreference", [
            ("TextBefore", ""),
            ("IncludePath", "false"),
            ("IncludeExtension", "false"),
            ("TextAfter", ""),
        ])
    elif v.kind == 0xCAA8:
        x.empty("PageNumberVariablePreference", [
            ("TextBefore", ""),
            ("Format", "Current"),
            ("TextAfter", ""),
            ("Scope", "SectionScope"),
        ])
    elif v.kind == 0xCAC0:
        x.empty("CaptionMetadataVariablePreference", [
            ("TextBefore", ""),
            ("MetadataProviderName", "$ID/#LinkInfoNameStr"),
            ("TextAfter", ""),
        ])


def on_master(w, story):
    """Whether a text frame of a master spread shows story ``story``."""
    def has(items):
        for i in items:
            if i.kind[0] == "text_frame" \
                    and i.kind[1].get("story") == story:
                return True
            if has(i.children):
                return True
        return False

    return any(has(s.items) for s in w.doc.master_spreads)


def variable_instance(w, x, v, story):
    """A text variable instance in place of its U+0018."""
    name = variable_name(v.name)
    x.start("TextVariableInstance") \
        .attr("Self", uref(v.uid)) \
        .attr("Name", name)
    # The displayed text is not stored. A file name variable with the
    # sample settings shows the document's name without extension; a
    # running header on a master page shows the variable's name, and a
    # chapter number in the current format the document's number, each
    # between the text before and after.
    definition = next((d for d in w.doc.text_variables if d.name == v.name),
                      None)
    result = None
    if definition is not None:
        def around(t):
            if definition.settings is None:
                return None
            s = definition.settings
            return f"{s.before}{t}{s.after}"

        if definition.kind == 0xCAA6 and definition.as_in_samples:
            result = w.name[:-5] if w.name.endswith(".indd") else w.name
        elif definition.kind == 0xCAAA and on_master(w, story):
            result = around(f"<{definition.name}>")
        elif definition.kind == 0xCAA9 and definition.settings is not None \
                and definition.settings.a == 0:
            pref = next((p for p in w.doc.prefs.values
                         if p.element == "ChapterNumberPreference"
                         and p.name == "ChapterNumber"), None)
            if pref is not None:
                result = around(pref.value)
    if result is not None:
        x.attr("ResultText", result)
    x.attr("AssociatedTextVariable", f"dTextVariablen{name}")
    x.end()


def topic_self(parent, t):
    """The ``Self`` of a topic: its parent's ``Self``, ``Topicn`` and
    its name."""
    return f"{parent}Topicn{t.name}"


def index_xml(x, idx):
    """The ``Index`` element and its topics."""
    def topic(parent, t):
        me = topic_self(parent, t)
        x.start("Topic") \
            .attr("Self", me) \
            .attr("SortOrder", t.sort_order) \
            .attr("Name", t.name)
        for c in t.children:
            topic(me, c)
        x.end()

    me = uref(idx.uid)
    x.start("Index").attr("Self", me)
    for t in idx.topics:
        topic(me, t)
    x.end()


def topic_refs(doc):
    """The ``Self`` of the topic of each page reference, by page
    reference UID."""
    out = {}

    def walk(parent, t):
        me = topic_self(parent, t)
        for r in t.refs:
            out[r] = me
        for c in t.children:
            walk(me, c)

    if doc.index is not None:
        me = uref(doc.index.uid)
        for t in doc.index.topics:
            walk(me, t)
    return out


def custom_mojikumi(x, m):
    """``BasedOnMojikumiSet`` and the spacing entries of a custom
    mojikumi table. A code without a known table name leaves the
    attribute out."""
    from .idml_attrs import builtin_cjk_table

    if m.based_on == 0:
        x.attr("BasedOnMojikumiSet", "Nothing")
    else:
        based = builtin_cjk_table(f"kMojikumiDefaultName{m.based_on}")
        if based is not None:
            x.attr("BasedOnMojikumiSet", based)
    x.start("Properties").start("OverrideMojikumiAkiList")
    for e in m.entries:
        x.empty("OverrideMojikumiAkiType", [
            ("TargetMojikumiClass", str(e.target_class)),
            ("SideMojikumiClass", str(e.side_class)),
            ("SideIsAfterTarget", str(e.side_is_after_target).lower()),
            ("Minimum", num(e.minimum)),
            ("Desired", num(e.desired)),
            ("Maximum", num(e.maximum)),
            ("CompressionPriority", str(e.compression_priority)),
            ("AkiDoesNotFloat", str(e.aki_does_not_float).lower()),
        ])
    x.end()
    x.end()


def toc_separator(s):
    """A TOC entry separator as IDML writes it: ``$ID/``, then the
    text with tab, U+0008, line feed, em space and em dash as ``^t``,
    ``^y``, ``^n``, ``^m`` and ``^_``."""
    out = ["$ID/"]
    for ch in s:
        if ch == "\t":
            out.append("^t")
        elif ch == "\u0008":
            out.append("^y")
        elif ch == "\n":
            out.append("^n")
        elif ch == "\u2003":
            out.append("^m")
        elif ch == "\u2014":
            out.append("^_")
        else:
            out.append(ch)
    return "".join(out)
