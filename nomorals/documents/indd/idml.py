"""IDML package writer: the Writer and the package assembly.

A faithful port of ``src/idml/mod.rs`` (plus ``pages.rs``) from
storytold/indd-utils (commit 136f973, dual MIT/Apache-2.0 — see
attribution.py). The part writers live in the ``idml_*`` sibling
modules and take the Writer as their first argument, as upstream's
``impl Writer`` blocks in the child modules do.
"""

from __future__ import annotations

from .idml_values import element_attrs
from .idml_xml import ZipWriter, self_name, style_path, uref
from .ids import Numbering
from .object import builtin_key

PACKAGING_NS = "http://ns.adobe.com/AdobeInDesign/idml/1.0/packaging"
MIMETYPE = "application/vnd.adobe.indesign-idml-package"

CONTAINER_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
    '\t<rootfiles>\n'
    '\t\t<rootfile full-path="designmap.xml" media-type="text/xml">\n'
    '\t\t</rootfile>\n'
    '\t</rootfiles>\n'
    '</container>'
)


# --- pages.rs ----------------------------------------------------------

def document_pages(doc):
    """UIDs of the document pages, in order."""
    return [p.uid for s in doc.spreads for p in s.pages]


def section_ranges(doc):
    """Sections in page order, each with the index of its first page
    and its length in pages. A section whose first page is not a
    document page is left out."""
    pages = document_pages(doc)
    if not pages:
        return []
    starts = []
    for s in doc.sections:
        if s.page is None:
            starts.append((0, s))
        elif s.page in pages:
            starts.append((pages.index(s.page), s))
    starts.sort(key=lambda t: t[0])
    deduped = []
    for i, s in starts:
        if not deduped or deduped[-1][0] != i:
            deduped.append((i, s))
    out = []
    for k, (i, s) in enumerate(deduped):
        end = deduped[k + 1][0] if k + 1 < len(deduped) else len(pages)
        out.append((s, i, end - i))
    return out


def alternate_layout_name(s):
    """The IDML ``AlternateLayout`` of a section: its name, with
    ``$ID/`` before a built-in key."""
    if s.alternate_layout is None:
        return None
    flag, name = s.alternate_layout
    return builtin_key(name) if flag == 1 else name


def alternate_layouts(doc):
    """For each section of ``section_ranges``, the number of pages from
    its start to the start of the next alternate layout (a section with
    a layout name) or the end of the document; and for each document
    page, the section that starts its alternate layout."""
    ranges = section_ranges(doc)
    total = sum(n for _s, _i, n in ranges)
    starts = [(i, s.uid) for s, i, _n in ranges
              if (lambda n: n is not None and n != "" and n != "$ID/")(
                  alternate_layout_name(s))]
    lengths = []
    for _s, i, _n in ranges:
        nxt = next((j for j, _u in starts if j > i), total)
        lengths.append(nxt - i)
    first = ranges[0][0].uid if ranges else None
    pages = []
    for p in range(total):
        found = None
        for j, u in reversed(starts):
            if j <= p:
                found = u
                break
        pages.append(found if found is not None else first)
    return lengths, pages


def number_style(code):
    """IDML ``PageNumberStyle`` of a section's style code."""
    return {
        Numbering.ARABIC: "Arabic",
        Numbering.LOWER_ROMAN: "LowerRoman",
        Numbering.KANJI: "Kanji",
        Numbering.DOUBLE_LEADING_ZEROS: "DoubleLeadingZeros",
        Numbering.UPPER_ROMAN: "UpperRoman",
        Numbering.LOWER_LETTERS: "LowerLetters",
        Numbering.SINGLE_LEADING_ZEROS: "SingleLeadingZeros",
    }.get(code)


def lower_roman(n):
    """Lower-case Roman numeral of ``n`` (1 or more)."""
    digits = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
              (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
              (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = []
    for value, text in digits:
        while n >= value:
            out.append(text)
            n -= value
    return "".join(out)


def upper_roman(n):
    return lower_roman(n).upper()


def single_leading_zeros(n):
    return f"{n:02}"


def double_leading_zeros(n):
    return f"{n:03}"


def lower_letter(n):
    """The letter of ``n`` from 1 to 26; other numbers as Arabic
    numbers, as no sample shows them."""
    if 1 <= n <= 26:
        return chr(ord("a") + n - 1)
    return str(n)


def kanji_digits(n):
    """Chinese numeral of ``n``, digit by digit (10 is 一〇), as the
    folios of a sample with that style show."""
    digits = ["〇", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    return "".join(digits[int(b) - ord("0")] for b in str(n))


def page_numbers(doc):
    """The number shown on each document page, from the sections. The
    section at the first page starts at its start number even when it
    continues numbering (from the previous document of a book). Pages
    not covered by a section are numbered by their position."""
    count = len(document_pages(doc))
    out = list(range(1, count + 1))
    nxt = None
    for s, first, length in section_ranges(doc):
        if nxt is not None and s.continue_numbering:
            number = nxt
        elif nxt is None and s.continue_numbering and first > 0:
            number = first + 1
        else:
            number = s.start
        for i in range(first, first + length):
            out[i] = number
            number += 1
        nxt = number
    return out


def page_names(doc):
    """The name of each document page: its number in its section's
    style, after the section prefix if the section includes it."""
    numbers = page_numbers(doc)
    out = [str(n) for n in numbers]
    for s, first, length in section_ranges(doc):
        if s.style == Numbering.LOWER_ROMAN:
            name = lower_roman
        elif s.style == Numbering.KANJI:
            name = kanji_digits
        elif s.style == Numbering.UPPER_ROMAN:
            name = upper_roman
        elif s.style == Numbering.SINGLE_LEADING_ZEROS:
            name = single_leading_zeros
        elif s.style == Numbering.DOUBLE_LEADING_ZEROS:
            name = double_leading_zeros
        elif s.style == Numbering.LOWER_LETTERS:
            name = lower_letter
        else:
            name = str
        for i in range(first, first + length):
            if numbers[i] > 0:
                out[i] = name(numbers[i])
                if s.include_prefix:
                    out[i] = s.prefix + out[i]
    return out


def page_sections(doc):
    """The section and page number of each document page."""
    numbers = page_numbers(doc)
    out = []
    for s, first, length in section_ranges(doc):
        for n in numbers[first:first + length]:
            out.append((s, n))
    return out


# --- Writer construction helpers ---------------------------------------

def group_paths(doc):
    """Names of the enclosing style groups of each style or group UID
    (styles.rs)."""
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


def topic_refs(doc):
    """The ``Self`` of the index topic of each page reference UID
    (designmap.rs)."""
    out = {}

    def topic_self(parent, t):
        return f"{parent}Topicn{t.name}"

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


def dom_version_of(doc):
    """The ``DOMVersion`` of the application that saved the document
    last, by the save history, else by the header. Several releases
    keep the DOM version of an earlier one."""
    header = (doc.version.major, doc.version.minor)
    major, minor = header
    if doc.last_session_version is not None \
            and doc.last_session_version[0] == header[0]:
        major, minor = doc.last_session_version
    if (major, minor) in [(9, 1), (9, 2), (11, 1), (14, 3),
                          (17, 1), (17, 2), (17, 3), (17, 4),
                          (18, 1), (18, 2), (18, 3), (18, 4),
                          (19, 1), (19, 2), (20, 1)]:
        minor = 0
    elif (major, minor) in [(16, 3), (16, 4)]:
        minor = 2
    elif (major, minor) == (20, 5):
        minor = 4
    return f"{major}.{minor}"


# --- Writer ------------------------------------------------------------

class Edition:
    ENGLISH = "english"
    FRENCH = "french"
    GERMAN = "german"
    DUTCH = "dutch"
    ITALIAN = "italian"
    KOREAN = "korean"
    JAPANESE = "japanese"
    CHINESE = "chinese"
    CZECH = "czech"
    PORTUGUESE = "portuguese"
    RUSSIAN = "russian"
    OTHER = "other"

    _UNKNOWN_USER = {
        "english": ["Unknown User Name"],
        "korean": ["알 수 없는 사용자 이름"],
        "japanese": ["不明なユーザー名"],
        "chinese": ["不明的使用者名稱", "未知用户名"],
        "german": ["Unbekannter Benutzername"],
        "french": ["Nom d'utilisateur inconnu"],
        "italian": ["Nome ut. sconosciuto"],
        "dutch": ["Onbekende gebruikersnaam"],
        "czech": ["Neznámé jméno uživatele"],
        "portuguese": ["Nome de usuário desconhecido"],
        "russian": ["Неизвестное имя пользователя"],
        "other": [],
    }

    _BEHAVIOR_NAMES = {
        ("english", "GoTo Anchor"): "Go To Destination",
        ("korean", "GoTo Anchor"): "대상으로 이동",
        ("korean", "GoTo First Page"): "첫 페이지로 이동",
        ("korean", "GoTo Next Page"): "다음 페이지로 이동",
        ("korean", "GoTo Previous Page"): "이전 페이지로 이동",
        ("korean", "GoTo Next View"): "다음 보기로 이동",
        ("korean", "SwitchToMSOState"): "상태로 이동",
        ("korean", "SwitchToNextMSOState"): "다음 상태로 이동",
        ("korean", "SwitchToPrevMSOState"): "이전 상태로 이동",
        ("chinese", "GoTo First Page"): "跳至第一頁",
        ("chinese", "GoTo Next Page"): "跳至下一頁",
        ("chinese", "GoTo Previous Page"): "上一頁",
        ("chinese", "GoTo Page"): "跳至頁面",
        ("chinese", "GoTo URL"): "前往 URL",
        ("chinese", "SwitchToMSOState"): "跳至狀態",
        ("chinese", "SwitchToNextMSOState"): "跳至下一個狀態",
        ("chinese", "SwitchToPrevMSOState"): "跳至上一個狀態",
        ("chinese", "Movie"): "影片",
        ("chinese", "Sound"): "聲音",
        ("chinese", "Show/Hide Fields"): "顯示/隱藏欄位",
        ("chinese", "Open File"): "開啟檔案",
        ("chinese", "Submit Form"): "送出表格",
        ("chinese", "Reset Form"): "清除表格",
        ("chinese", "Print Form"): "列印表格",
    }

    @classmethod
    def unknown_user(cls, edition):
        """The name this edition gives an unknown document user;
        Chinese documents show two."""
        return cls._UNKNOWN_USER.get(edition, [])

    @classmethod
    def behavior_name(cls, edition, key):
        """The name this edition gives a behaviour with the built-in
        name ``key``, where a sample shows it."""
        if not key.startswith("$$$/Forms Action/"):
            return None
        return cls._BEHAVIOR_NAMES.get(
            (edition, key[len("$$$/Forms Action/"):]))


class Writer:
    """The IDML writer: the document, its DOM version, and the derived
    state the part writers share."""

    def __init__(self, doc, name):
        self.doc = doc
        self.dom = dom_version_of(doc)
        # Document file name.
        self.name = name
        # Names of the enclosing style groups of each style or group UID.
        self.group_path = group_paths(doc)
        # Values left out while writing, reported with the model's
        # warnings.
        self.warnings = []
        # For each document page, the section that starts its alternate
        # layout (`alternate_layouts`).
        self.page_layouts = alternate_layouts(doc)[1]
        # For each document page, its section and page number.
        self.page_sections = page_sections(doc)
        # The `Self` of the index topic of each page reference UID.
        self.topics = topic_refs(doc)
        # The anchored object settings of the anchored items being
        # written, innermost last: items inside them carry the settings
        # too.
        self.anchors = []
        # UIDs a `TimingTarget` can refer to (`timing_targets`).
        self._timing_targets = None
        # UIDs of the movies and sounds written (`written_media`).
        self._media_uids = None
        # Observed attributes by element path (`values::element_attrs`).
        self._observed = {}
        # Effective values of object styles, by UID (`applied`).
        self.style_values = {}

    def warn(self, message):
        self.warnings.append(message)

    def dom_version(self):
        """The ``DOMVersion`` written, as (major, minor)."""
        parts = self.dom.split(".")
        try:
            major = int(parts[0])
        except (IndexError, ValueError):
            major = 0
        try:
            minor = int(parts[1])
        except (IndexError, ValueError):
            minor = 0
        return major, minor

    def saved_by(self, v):
        """Whether the document's version, or the application version
        of its last session, is at least ``v``. IDML follows the
        exporting application, which can be later than the version in
        the header."""
        d = self.doc.version
        return (d.major, d.minor) >= v or (
            self.doc.last_session_version is not None
            and self.doc.last_session_version >= v)

    def edition(self):
        """The edition that made the document, by the language of the
        name of its first assignment."""
        name = self.doc.assignment_name
        if name is None:
            return None
        return {
            "Unassigned InCopy Content": Edition.ENGLISH,
            "Contenu InCopy non affecté": Edition.FRENCH,
            "Nicht zugewiesener InCopy-Inhalt": Edition.GERMAN,
            "Niet toegewezen InCopy-inhoud": Edition.DUTCH,
            "Contenuto InCopy non assegnato": Edition.ITALIAN,
            "할당되지 않은 InCopy 내용": Edition.KOREAN,
            "アサインされていない InCopy の内容": Edition.JAPANESE,
            "未指定的 InCopy 內容": Edition.CHINESE,
            "未指定的 InCopy 内容": Edition.CHINESE,
        }.get(name, Edition.OTHER)

    def saving_edition(self):
        """The edition that saved the document, by the black name of
        the XMP packet."""
        name = self.doc.black_name
        if name is None:
            return None
        return {
            "Black": Edition.ENGLISH,
            "검정": Edition.KOREAN,
            "黒": Edition.JAPANESE,
            "黑色": Edition.CHINESE,
            "Schwarz": Edition.GERMAN,
            "Noir": Edition.FRENCH,
            "Nero": Edition.ITALIAN,
            "Zwart": Edition.DUTCH,
            "Černá": Edition.CZECH,
            "Preto": Edition.PORTUGUESE,
            "Черный": Edition.RUSSIAN,
        }.get(name, Edition.OTHER)

    def japanese_session(self):
        """Whether the last session in the save history was one of a
        Japanese or Chinese edition (code 0x0101); ``None`` when the
        history cannot be read."""
        if self.doc.last_session_code is None:
            return None
        return self.doc.last_session_code == 0x0101

    def observed(self, path):
        """The attributes every IDML of the document's version has on
        the elements on ``path``."""
        if path not in self._observed:
            self._observed[path] = element_attrs(path,
                                                 self.doc.version.major)
        return self._observed[path]

    def package_root(self, x, kind):
        x.start(f"idPkg:{kind}") \
            .attr("xmlns:idPkg", PACKAGING_NS) \
            .attr("DOMVersion", self.dom)

    def grouped_name(self, uid, name, builtin):
        """The IDML ``Name`` of a style: the names of its style groups
        and its own, joined by ``:``, with a ``:`` inside a name
        written ``\\:``, and ``$ID/`` first for a built-in style. Its
        ``Self`` is this name escaped with ``self_name``."""
        groups = self.group_path.get(uid, [])
        joined = style_path(list(groups) + [name])
        if builtin:
            return builtin_key(joined)
        return joined

    def family_names(self, f):
        """The name IDML gives a font family, and the name its fonts'
        names are made from: the native name when the family's writing
        script is the script of the document users, and ``$ID/`` first
        for a built-in name."""
        native = (self.doc.users_script is not None
                  and self.doc.users_script != 0
                  and self.doc.users_script == f.writing_script
                  and f.native_name
                  and f.native_name != f.name)
        base = f.native_name if native else f.name
        idml = builtin_key(base) if f.builtin else base
        return idml, base

    def composite_font_of(self, f):
        """The composite font a font family stands for: the one whose
        name is the family's name, with ``<hhhh>`` escapes decoded."""
        if not self.doc.composite_fonts:
            return None
        name = f.unescaped_name()
        for c in self.doc.composite_fonts:
            if c.name.name == name:
                return c
        return None

    def style_ref(self, uid, paragraph):
        prefix = "ParagraphStyle" if paragraph else "CharacterStyle"
        style = self.doc.styles.get(uid) if uid is not None else None
        if style is not None:
            return (f"{prefix}/"
                    f"{self_name(self.grouped_name(style.uid, style.name, style.builtin))}")
        if paragraph:
            return "ParagraphStyle/$ID/NormalParagraphStyle"
        return "CharacterStyle/$ID/[No character style]"

    def table_style_name(self, s):
        return self.grouped_name(s.uid, s.name, s.builtin)

    def table_style_ref(self, tag, s):
        """``CellStyle/...`` or ``TableStyle/...`` reference of a
        style."""
        return f"{tag}/{self_name(self.table_style_name(s))}"

    def cell_style_ref(self, uid):
        """The reference for an applied cell style UID; 0 is
        ``[None]``."""
        if uid == 0:
            return "CellStyle/$ID/[None]"
        s = self.doc.cell_styles.get(uid)
        if s is None:
            return None
        return self.table_style_ref("CellStyle", s)

    def written_page_items(self):
        """Every page item the package writes, by UID: spread and
        master spread items, anchored items, and items in table
        cells."""
        out = {}
        stack = []
        for s in list(self.doc.spreads) + list(self.doc.master_spreads):
            stack.extend(s.items)
        for story in self.doc.stories:
            for items in story.anchors.values():
                stack.extend(items)
            for t in story.tables.values():
                for c in t.cells:
                    stack.extend(c.items)
        while stack:
            item = stack.pop()
            if item.uid in out:
                continue
            out[item.uid] = item
            stack.extend(item.children)
        return out

    def written_media(self):
        """UIDs of the movies and sounds written."""
        if self._media_uids is None:
            out = set()
            stack = list(self.written_page_items().values())
            while stack:
                item = stack.pop()
                out.update(m.uid for m in item.media)
                if item.kind[0] == "form":
                    for state in item.kind[1].states:
                        for i in state.items:
                            stack.append(i)
                            stack.extend(i.children)
            self._media_uids = out
        return self._media_uids

    def timing_targets(self):
        """UIDs a ``TimingTarget`` can refer to: the page items,
        movies and sounds written and the behaviours of the buttons
        among the items."""
        if self._timing_targets is None:
            out = set(self.written_media())
            for item in self.written_page_items().values():
                out.add(item.uid)
                if item.kind[0] == "form" \
                        and item.kind[1].kind == "button":
                    out.update(b.uid for b in item.kind[1].behaviors)
            self._timing_targets = out
        return self._timing_targets


def swatch_refs(parts):
    """The ``Color/u…`` and ``Gradient/u…`` references in the package
    parts: an attribute value or element text that is such a
    reference."""
    out = set()
    for part in parts:
        for prefix in ("Color/u", "Gradient/u", "PastedSmoothShade/u"):
            start = 0
            while True:
                i = part.find(prefix, start)
                if i < 0:
                    break
                start = i + 1
                if i > 0 and part[i - 1] not in ('"', ">"):
                    continue
                rest = part[i + len(prefix):]
                n = 0
                while n < len(rest) and rest[n] in "0123456789abcdefABCDEF":
                    n += 1
                if n > 0 and n < len(rest) and rest[n] in ('"', "<"):
                    out.add(part[i:i + len(prefix) + n])
    return out


def write(doc, name):
    """Write ``doc`` as an IDML package. ``name`` is the document name
    (file name). Returns ``(bytes, warnings)`` — warnings about values
    left out because the IDML schema does not allow them."""
    from .idml_designmap import designmap
    from .idml_resources import fonts, graphic, preferences
    from .idml_spread import spread
    from .idml_story import backing_story, story, tags
    from .idml_styles import styles

    w = Writer(doc, name)
    files = {}
    files["META-INF/container.xml"] = CONTAINER_XML
    files["designmap.xml"] = designmap(w, name)
    files["Resources/Fonts.xml"] = fonts(w)
    files["Resources/Styles.xml"] = styles(w)
    files["Resources/Preferences.xml"] = preferences(w)
    files["XML/BackingStory.xml"] = backing_story(w)
    files["XML/Tags.xml"] = tags(w)
    names = page_names(doc)
    unused = [0]
    for s in doc.master_spreads:
        files[f"MasterSpreads/MasterSpread_u{s.uid:x}.xml"] = \
            spread(w, s, True, unused, names)
    page_index = [0]
    for s in doc.spreads:
        files[f"Spreads/Spread_u{s.uid:x}.xml"] = \
            spread(w, s, False, page_index, names)
    for s in doc.stories:
        files[f"Stories/Story_u{s.uid:x}.xml"] = story(w, s)
    # Unnamed colours and gradients are written only where the other
    # parts refer to them, so the graphic part is written last.
    refs = swatch_refs(files.values())
    files["Resources/Graphic.xml"] = graphic(w, refs)
    z = ZipWriter()
    z.add("mimetype", MIMETYPE.encode("ascii"))
    for path in sorted(files):
        z.add(path, files[path].encode("utf-8"))
    z.finish()
    out = z.out
    return out.getvalue(), w.warnings
