"""Links, style mappings and import options of linked stories and of
stories placed from Word or Excel files.

A faithful port of ``src/idml/story_link.rs`` from storytold/indd-utils
(commit 136f973, dual MIT/Apache-2.0 — see attribution.py).
"""

from __future__ import annotations

from .idml_values import Node
from .idml_xml import write_properties


def _typed_list(tag, items):
    """A ``Properties`` child of type ``list`` holding one typed
    ``ListItem`` per value."""
    return Node(tag=tag, attrs=[("type", "list")], children=[
        Node(tag="ListItem", attrs=[("type", ty)], text=text)
        for ty, text in items])


def _style_clash(code):
    return {0: "ResolveClashUseExisting",
            2: "ResolveClashAutoRename"}.get(code)


def _word_items(s, major):
    """The items of ``WordRTFImportPreferences`` from the 19 u32 of
    chunk 0x12210: 20 items from version 14, 19 before. Items without
    a stored field have the value of every sample. ``None`` for an
    unknown code."""
    def flag(i):
        return ("boolean", str(s[i] != 0).lower())

    def constant(v):
        return ("boolean", str(v).lower())

    page_breaks = {0: "PageBreak", 2: "None"}.get(s[5])
    if page_breaks is None:
        return None
    clash11 = _style_clash(s[11])
    clash12 = _style_clash(s[12])
    if clash11 is None or clash12 is None:
        return None
    items = [
        flag(0), flag(1), ("enumeration", page_breaks), constant(True),
        ("enumeration", "UnformattedTable"), flag(8), flag(9),
        constant(False), ("enumeration", clash11),
        ("enumeration", clash12), flag(14), flag(15), flag(16), flag(17),
    ]
    if major >= 14:
        items.append(constant(False))
    items.extend([constant(True), constant(True), ("long", str(s[4])),
                  constant(False), constant(False)])
    return items


def _excel_items(e):
    """The items of ``ExcelImportPreferences``. Items without a stored
    field have the value of every sample. ``None`` for an unknown table
    code."""
    table = {0: "ExcelFormattedTable", 1: "ExcelUnformattedTable"}.get(e.table)
    if table is None:
        return None
    return [
        ("enumeration", "Spreadsheet"),
        ("long", "3"),
        ("boolean", "false"),
        ("enumeration", table),
        ("boolean", "true"),
        ("boolean", "true"),
        ("string", e.view),
        ("string", e.sheet),
        ("string", e.range),
        ("long", "1"),
    ]


def _group(name, group):
    """A style mapping name: a group name with ``:*`` after it."""
    return f"{name}:*" if group else name


def story_link_properties(w, x, s):
    """``<Properties>`` of a story placed from a Word or Excel file:
    its import options, then its style mapping preferences."""
    linked = s.settings.linked
    nodes = []
    if linked is not None:
        imp = linked.import_options
        if imp is not None:
            if imp[0] == "word":
                items = _word_items(imp[1], w.doc.version.major)
                if items is None:
                    w.warn(f"story {s.uid}: Word import options with an "
                           f"unknown code left out")
                else:
                    nodes.append(_typed_list("WordRTFImportPreferences", items))
            elif imp[0] == "excel":
                items = _excel_items(imp[1])
                if items is None:
                    w.warn(f"story {s.uid}: Excel import options with an "
                           f"unknown code left out")
                else:
                    nodes.append(_typed_list("ExcelImportPreferences", items))
        prefs = linked.mapping_prefs
        if prefs is not None:
            paragraph, character = prefs
            if paragraph or character:
                def lst(names):
                    return _typed_list("ListItem",
                                       [("string", n) for n in names])

                nodes.append(Node(tag="StyleMappingPreferences",
                                  attrs=[("type", "list")],
                                  children=[lst(paragraph),
                                            lst(character)]))
    write_properties(x, [], nodes)


def linked_story_option(x, s):
    """``LinkedStoryOption`` of a story with a link and chunk 0x21D12.
    The first three values have no identified field (constant in every
    sample)."""
    linked = s.settings.linked
    if linked is not None and linked.apply_style_mappings is not None \
            and linked.link is not None:
        x.empty("LinkedStoryOption", [
            ("UpdateWhileSaving", "false"),
            ("WarnOnUpdateOfEditedStory", "true"),
            ("RemoveForcedLineBreaks", "false"),
            ("ApplyStyleMappings", str(linked.apply_style_mappings).lower()),
        ])


def style_mappings(w, x, owner, m):
    """``ParaStyleMapping`` and ``CharStyleMapping`` elements of the
    story or document whose ``Self`` is ``owner``. A mapping with an
    unknown rule and the lists whose element is not known are left out
    with a warning."""
    for tag, lst in [("ParaStyleMapping", m.paragraph),
                     ("CharStyleMapping", m.character)]:
        for i, e in enumerate(lst):
            # A group name ends in `:*`.
            if e.rule == 0:
                rule, sg, dg = "StyleMappingRule", False, False
            elif e.rule == 1:
                rule, sg, dg = "GroupMappingRule", True, True
            elif e.rule == 2:
                rule, sg, dg = "StyleToGroupMappingRule", False, True
            else:
                w.warn(f"{owner}: {tag} {i} with rule {e.rule} left out")
                continue
            x.empty(tag, [
                ("Self", f"{owner}{tag}{i}"),
                ("SourceStyleName", _group(e.source, sg)),
                ("DestinationStyleName", _group(e.destination, dg)),
                ("MappingRuleType", rule),
            ])
    if m.other > 0:
        w.warn(f"{owner}: {m.other} table or cell style mappings left out")
