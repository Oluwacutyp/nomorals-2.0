"""Measured fidelity: paired-file diffing of IDML packages, with value
coverage as the headline number.

A faithful port of ``tools/compare.py`` (``load``, ``text_ranges``,
``story_text``, ``norm``, ``canon``, ``props``, ``values``,
``value_coverage``) from storytold/indd-utils (commit 136f973, dual
MIT/Apache-2.0 — see attribution.py). The method is the one
``docs/measurement.md`` defines: the values of a reference package are
element presence (``Self`` plus the first attributed child), attributes
(other than ``Self``, numbers to 6 significant digits), ``<Properties>``
children, story text (identical including paragraph breaks) and text
ranges (offset and flow, in UTF-16 units); value coverage is the share
of the reference's values the other package reproduces exactly, and
extra values are counted the other direction.
"""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import io
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

from .errors import InddError

#: Elements that wrap part of a story's text without being text
#: themselves. IDML puts them around ranges (ParagraphStyleRange >
#: HyperlinkTextSource > CharacterStyleRange, XMLElement >
#: ParagraphStyleRange) or inside a range around text; text_ranges
#: looks through them.
WRAPPERS = {"XMLElement", "HyperlinkTextSource", "CrossReferenceSource",
            "Change", "EndnoteRange"}
#: Children that are neither text nor a character position.
NOT_TEXT = {"Properties", "XMLAttribute"}
#: Elements that hold text of their own inside a story: their ranges
#: are counted in a text flow of their own.
NESTED = {"Footnote", "Cell", "Note"}
#: Children of a child element with attributes that are compared as
#: "<parent>/<child>/<grandchild>".
NESTED_OPTIONS = {("TextWrapPreference", "ContourOption")}

#: Gap keys for values that could not be compared one by one.
NO_ELEMENT = "(element not produced)"
NO_TEXT = "(story text differs)"
NOT_IN_REFERENCE = "(element not in reference)"
ELEMENT = "(element)"

#: Values left out of value coverage because they cannot be derived
#: from an INDD file: (tag pattern, attribute or P.<Properties child>
#: pattern, reason), patterns as in fnmatch. docs/measurement.md
#: explains each entry.
EXCLUDED = [
    ("Font", "Status",
     "whether the font is installed on the exporting computer"),
    ("Document/{http://ns.adobe.com/AdobeInDesign/idml/1.0/packaging}*",
     "src", "file names of the package parts, chosen by the IDML writer"),
    ("DocumentUser", "P.UserColor",
     "colour the exporting InDesign gives each user, not the colour in the INDD"),
    ("Assignment", "Name",
     "the exporting InDesign's name for unassigned content, which depends on its language"),
    ("Link", "LinkImportTime",
     "local time of the exporting computer; the INDD stores UTC"),
    ("Link", "LinkImportModificationTime",
     "local time of the exporting computer; the INDD stores UTC"),
    ("*", "BitmapPrinting",
     "evaluated by the exporting computer; not stored in the INDD"),
] + [
    ("Font", k,
     "axes of the font file on the exporting computer; not stored in the INDD")
    for k in ("NumDesignAxes", "DesignAxesName", "DesignAxesValues",
              "P.DesignAxesRange")
] + [
    ("*", k, "written by a plugin of the exporting InDesign; not in the IDML schema")
    for k in ("Observers", "Subjects", "DisplayErrorDialogs", "EventFilter",
              "JavaScript", "SubjectScriptTagFilter", "UseDebugger",
              "CompactedDataStore", "ComponentId")
] + [
    ("Story", "P.DocsFlowImportSettings",
     "written by a plugin of the exporting InDesign; not in the IDML schema"),
    ("Story", "P.WordsFlow*",
     "written by a plugin of the exporting InDesign; not in the IDML schema"),
    # ELEMENT leaves out the whole element, with all its values.
    ("Document/WordsFlowPreference", ELEMENT,
     "written by a plugin of the exporting InDesign; not in the IDML schema"),
]

#: Attributes of a Link that describe the linked file as InDesign
#: found it when it exported the IDML. When the IDML's LinkResourceURI
#: differs from the converter's (which comes from the INDD), the link
#: was resolved to another file on the exporting computer, and these
#: values of that link are left out.
RELINKED = re.compile(r"LinkResource.*|LinkImport.*")

#: Font technology suffix that IDML adds to some family names,
#: depending on the fonts installed on the exporting computer.
FONT_SUFFIX = re.compile(r" \((OTF|TT|T1)\)$")


@dataclass
class FidelityResult:
    """The result of diffing one IDML package against a reference."""

    #: Share of the reference's values reproduced exactly.
    coverage: float = 1.0
    #: Reference values not reproduced, each named by element and
    #: attribute (a wrong value names what it should have been).
    missing: list = field(default_factory=list)
    #: Values the package has and the reference does not, each named.
    extra: list = field(default_factory=list)
    #: Values the reference holds.
    total: int = 0
    #: Values of the reference reproduced exactly.
    reproduced: int = 0

    @property
    def headline(self):
        """``"<reproduced>/<total> values, <coverage:.1%> coverage"``."""
        return (f"{self.reproduced}/{self.total} values, "
                f"{self.coverage:.1%} coverage")


def _open(package):
    if isinstance(package, (bytes, bytearray)):
        return zipfile.ZipFile(io.BytesIO(bytes(package)))
    return zipfile.ZipFile(package)


def text_ranges(story):
    """Map (text flow, offset) -> attributes of the paragraph and
    character range starting there, with keys prefixed PSR. and CSR.
    Flow 0 is the story's own text; each footnote, table cell and note
    is a flow of its own, numbered in document order. Ranges inside
    wrapper elements count, at whatever depth. Offsets count Content
    in UTF-16 code units and every other element inside a character
    range (Br, Table, Footnote, an anchored frame) as one position;
    elements outside character ranges (StoryPreference, ...) are not
    text."""
    out = {}
    flows = 0

    def walk(el, pa, in_csr, flow, pos):
        """Walk the children of ``el`` inside the paragraph attributes
        ``pa``; ``in_csr`` tells whether ``el`` is inside a character
        range. Returns the offset after them."""
        for ch in el:
            if ch.tag in NOT_TEXT:
                continue
            if ch.tag == "ParagraphStyleRange":
                pos = walk(ch, {"PSR." + k: v
                                for k, v in props(ch).items()},
                           False, flow, pos)
            elif ch.tag == "CharacterStyleRange":
                start = pos
                pos = walk(ch, pa, True, flow, pos)
                if pos > start:
                    ca = {"CSR." + k: v for k, v in props(ch).items()}
                    # A range nested at the same offset (Change > range
                    # at the start of this one) keeps the offset.
                    out.setdefault((flow, start), {**pa, **ca})
            elif ch.tag in WRAPPERS:
                pos = walk(ch, pa, in_csr, flow, pos)
            elif ch.tag == "Content":
                pos += len((ch.text or "").encode("utf-16-le")) // 2
            else:
                if in_csr:
                    pos += 1
                nested(ch)
        return pos

    def nested(el):
        """Walk the text flows held in ``el`` (a table's cells, a
        footnote)."""
        nonlocal flows
        if el.tag in NESTED:
            flows += 1
            walk(el, {}, False, flows, 0)
            return
        for ch in el:
            if ch.tag != "Properties":
                nested(ch)

    walk(story, {}, False, 0, 0)
    return out


def nested_options(elements, tag, s, el):
    for g in el:
        if (el.tag, g.tag) in NESTED_OPTIONS and len(g.attrib):
            elements.setdefault((f"{tag}/{g.tag}", s), g)


def story_text(story):
    parts = []
    for el in story.iter():
        if el.tag == "Content":
            parts.append(el.text or "")
        elif el.tag == "Br":
            parts.append("\n")
    return "".join(parts)


def load(package):
    """Map Self -> element, plus story texts and text ranges, from an
    IDML package (bytes or a path)."""
    z = _open(package)
    elements = {}
    stories = {}
    ranges = {}
    for name in z.namelist():
        if not name.endswith(".xml"):
            continue
        try:
            root = ET.fromstring(z.read(name))
        except ET.ParseError as exc:
            raise InddError.corrupt(f"{name}: {exc}") from exc
        for el in root.iter():
            s = el.get("Self")
            if s is not None:
                if el.tag == "Font":
                    s = font_self(el, s)
                elements.setdefault((el.tag, s), el)
                # Children without Self (TextFramePreference, ...) are
                # compared as "<parent tag>/<child tag>" of the parent.
                for ch in el:
                    if ch.get("Self") is not None or ch.tag == "Properties":
                        continue
                    if len(ch.attrib):
                        elements.setdefault((f"{el.tag}/{ch.tag}", s), ch)
                        nested_options(elements, f"{el.tag}/{ch.tag}",
                                       s, ch)
                    else:
                        # A container without attributes
                        # (TransparencySetting): compare its children as
                        # "<parent>/<child>/<grandchild>".
                        for g in ch:
                            if g.get("Self") is None and len(g.attrib):
                                elements.setdefault(
                                    (f"{el.tag}/{ch.tag}/{g.tag}", s), g)
        if name == "Resources/Preferences.xml":
            for el in root:
                elements.setdefault((el.tag, "Preferences"), el)
                nested_options(elements, el.tag, "Preferences", el)
        if name.startswith("Stories/"):
            for st in root.iter("Story"):
                stories[st.get("Self")] = story_text(st)
                ranges[st.get("Self")] = text_ranges(st)
    return elements, stories, ranges


def norm(v):
    """Normalise numbers so 1 == 1.0 and tiny float noise is ignored."""

    def fix(tok):
        try:
            f = float(tok)
        except ValueError:
            return tok
        return f"{f:.6g}"

    return " ".join(fix(t) for t in v.split(" "))


def canon(el):
    """A structured property (TabList, AllNestedStyles, ...) as one
    string: each leaf as tag=text with numbers normalised, in document
    order."""
    if len(el) == 0 and el.get("type") != "list":
        return f"{el.tag}={norm(el.text or '')}"
    return el.tag + "[" + ",".join(canon(c) for c in el) + "]"


def props(el):
    """Attributes plus <Properties> children (as P.Name)."""
    a = dict(el.attrib)
    p = el.find("Properties")
    if p is not None:
        for c in p:
            if c.get("type") == "list":
                a["P." + c.tag] = canon(c)
            elif len(c) == 0:
                text = c.text or ""
                # A line feed and tabs alone are the indentation of an
                # element written open and closed.
                if re.fullmatch(r"\n\t*", text):
                    text = ""
                extra = sorted((k, v) for k, v in c.attrib.items()
                               if k != "type")
                if extra:
                    text = " ".join(f"{k}={v}" for k, v in extra)
                if c.tag == "Contents":
                    n = norm(text)
                    text = (f"md5:{hashlib.md5(n.encode()).hexdigest()} "
                            f"{len(n)}")
                a["P." + c.tag] = text
    return a


def font_name(tag, key, v, family=None):
    """A value with the font technology suffix removed if it names a
    font family: FontFamily Name, Font FontFamily, AppliedFont and
    BulletsFont. A Font's Name starts with its family (passed as
    family)."""
    if (tag, key) in (("FontFamily", "Name"), ("Font", "FontFamily")) or \
            key.endswith(("P.AppliedFont", "P.BulletsFont")):
        return FONT_SUFFIX.sub("", v)
    if (tag, key) == ("Font", "Name") and family \
            and v.startswith(family):
        return FONT_SUFFIX.sub("", family) + v[len(family):]
    return v


def font_self(el, s):
    """A Font's Self with the technology suffix removed from the
    family name it contains, so that fonts match whether or not the
    exporting computer added the suffix."""
    family = el.get("FontFamily") or ""
    base = FONT_SUFFIX.sub("", family)
    if base != family:
        return s.replace("Fontn" + family, "Fontn" + base, 1)
    return s


@functools.cache
def excluded(tag, key):
    return any(fnmatch.fnmatchcase(key, k)
               and fnmatch.fnmatchcase(tag, t)
               for t, k, _ in EXCLUDED)


def tree(el):
    """An element with its attributes, text and descendants as one
    string, numbers normalised: the value of a structured <Properties>
    child such as PathGeometry."""
    attrs = ",".join(f"{k}={norm(v)}"
                     for k, v in sorted(el.attrib.items()) if k != "type")
    kids = ",".join(tree(c) for c in el)
    return f"{el.tag}({attrs})[{kids}]{norm((el.text or '').strip())}"


def values(tag, el):
    """The values of an element that value coverage counts: attributes
    and <Properties> children, without Self and without EXCLUDED
    values."""
    a = props(el)
    p = el.find("Properties")
    if p is not None:
        for c in p:
            if c.get("type") != "list" and len(c):
                a["P." + c.tag] = tree(c)
    return {k: v for k, v in a.items()
            if k != "Self" and not excluded(tag, k)}


def measure_fidelity(reference, package):
    """Diff IDML package ``package`` against reference package
    ``reference`` (each bytes or a path) and report value coverage as
    the headline: the share of the reference's values the package
    reproduces exactly, with the uncovered values named and the extra
    values counted the other direction."""
    ref_el, ref_st, ref_rg = load(reference)
    our_el, our_st, our_rg = load(package)
    total = Counter()
    wrong = Counter()
    missing = Counter()
    extra = Counter()
    missing_lines = []
    extra_lines = []

    def compare(tag, theirs, mine, where):
        for k, v in theirs.items():
            key = (tag, k)
            total[key] += 1
            o = mine.get(k)
            if o is None:
                missing[key] += 1
                missing_lines.append(
                    f"{where} {tag} {k}: not produced "
                    f"(reference {v!r})")
            elif norm(font_name(tag, k, o,
                                mine.get("FontFamily"))) != \
                    norm(font_name(tag, k, v,
                                   theirs.get("FontFamily"))):
                wrong[key] += 1
                missing_lines.append(
                    f"{where} {tag} {k}: expected {v!r}, got {o!r}")
        for k in mine.keys() - theirs.keys():
            extra[(tag, k)] += 1
            extra_lines.append(f"{where} {tag} {k}: extra value "
                               f"{mine[k]!r}")

    for (tag, s), el in ref_el.items():
        if excluded(tag, ELEMENT):
            continue
        theirs = values(tag, el)
        mine = our_el.get((tag, s))
        if mine is None:
            n = 1 + len(theirs)
            total[(tag, NO_ELEMENT)] += n
            missing[(tag, NO_ELEMENT)] += n
            missing_lines.append(
                f"{tag} {s}: element not produced "
                f"({n} values)")
            continue
        total[(tag, ELEMENT)] += 1
        mine = values(tag, mine)
        if tag == "Link" and mine.get("LinkResourceURI") not in (
                None, theirs.get("LinkResourceURI")):
            theirs = {k: v for k, v in theirs.items()
                      if not RELINKED.fullmatch(k)}
            mine = {k: v for k, v in mine.items()
                    if not RELINKED.fullmatch(k)}
        compare(tag, theirs, mine, f"{tag} {s}")
    for (tag, s), el in our_el.items():
        if (tag, s) not in ref_el and not excluded(tag, ELEMENT):
            extra[(tag, NOT_IN_REFERENCE)] += 1 + len(values(tag, el))
            extra_lines.append(
                f"{tag} {s}: element not in reference "
                f"({1 + len(values(tag, el))} values)")
    for sid, text in ref_st.items():
        key = ("Story", "(text)")
        total[key] += 1
        if sid not in our_st:
            missing[key] += 1
            missing_lines.append(f"Story {sid}: text not produced")
        elif our_st[sid] != text:
            wrong[key] += 1
            missing_lines.append(f"Story {sid}: text differs")
    for sid in our_st.keys() - ref_st.keys():
        extra[("Story", "(text)")] += 1
        extra_lines.append(f"Story {sid}: extra story text")
        extra[("TextRange", NOT_IN_REFERENCE)] += sum(
            1 + len(a) for a in our_rg.get(sid, {}).values())
    for sid, rgs in ref_rg.items():
        if our_st.get(sid) != ref_st.get(sid):
            n = sum(1 + len(a) for a in rgs.values())
            total[("TextRange", NO_TEXT)] += n
            missing[("TextRange", NO_TEXT)] += n
            missing_lines.append(
                f"Story {sid}: text differs ({n} text-range values)")
            continue
        mine = our_rg.get(sid, {})
        for start, attrs in rgs.items():
            if start not in mine:
                n = 1 + len(attrs)
                total[("TextRange", NO_ELEMENT)] += n
                missing[("TextRange", NO_ELEMENT)] += n
                missing_lines.append(
                    f"Story {sid} range {start}: range not produced "
                    f"({n} values)")
                continue
            total[("TextRange", ELEMENT)] += 1
            compare("TextRange", attrs, mine[start],
                    f"Story {sid} range {start}")
        for start in mine.keys() - rgs.keys():
            extra[("TextRange", NOT_IN_REFERENCE)] += 1 + len(mine[start])
            extra_lines.append(
                f"Story {sid} range {start}: extra range "
                f"({1 + len(mine[start])} values)")
    n = sum(total.values())
    bad = sum(wrong.values()) + sum(missing.values())
    return FidelityResult(
        coverage=(n - bad) / n if n else 1.0,
        missing=missing_lines,
        extra=extra_lines,
        total=n,
        reproduced=n - bad,
    )
