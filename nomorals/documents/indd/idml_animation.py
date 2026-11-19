"""``MotionPreset``, ``TimingSetting`` and ``AnimationSetting``.

A faithful port of ``src/idml/animation.rs`` from storytold/indd-utils
(commit 136f973, dual MIT/Apache-2.0 — see attribution.py). Values
with one value in every sample and no stored field found
(``PlaysLoop``, ``ReverseAnimation``, ``InitiallyHidden``,
``HiddenAfter``, the preset locks, ``PathOpen``) are written as that
value.
"""

from __future__ import annotations

from .idml_xml import num, uref

#: ``TriggerEvent`` codes of timing lists.
TRIGGERS = [(1, "OnPageLoad"), (5, "OnClick"), (6, "OnRelease"),
            (7, "OnRollover")]
DESIGN_OPTIONS = [(0, "FromCurrentAppearance"), (2, "ToCurrentLocation")]
EASE_TYPES = [(0, "NoEase"), (4, "CustomEase")]
#: ``Properties`` children of an animation by stored property name, in
#: IDML order.
PROPERTY_ARRAYS = [
    ("Alpha_Amount", "OpacityArray"),
    ("Rotation_Z", "RotationArray"),
    ("Scale_X", "ScaleXArray"),
    ("Scale_Y", "ScaleYArray"),
]


def _code(table, v):
    for c, n in table:
        if c == v:
            return n
    return None


def _six_digits(v):
    """A number rounded to 6 significant digits, as IDML writes
    ``TransformOffsets``."""
    return num(float(f"{v:.5e}"))


def _point(x, y):
    return f"{num(x)} {num(y)}"


def _cdata_text(x, s):
    """CDATA sections holding ``s``, split inside any ``]]>``."""
    from .idml_xml import CDATA_SECTION, xml_char
    s = "".join(c for c in s if xml_char(c))
    parts = s.split("]]>")
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            x.cdata(f"{p}]]", CDATA_SECTION)
            x.cdata(">", CDATA_SECTION)
        elif p:
            x.cdata(p, CDATA_SECTION)


def motion_preset(x, p):
    """A ``MotionPreset`` of designmap.xml."""
    x.start("MotionPreset") \
        .attr("Self", uref(p.uid)) \
        .attr("Name", p.name.idml()) \
        .attr("EditLocked", "false") \
        .attr("DeleteLocked", "false") \
        .attr("NameLocked", "false")
    x.start("Properties").start("Contents")
    _cdata_text(x, p.contents)
    x.end().end().end()


def written_media(w):
    """UIDs of the movies and sounds written: in the page items
    written and in the states of form fields among them."""
    if w._media_uids is None:
        out = set()
        stack = list(w.written_page_items().values())
        while stack:
            item = stack.pop()
            out.update(m.uid for m in item.media)
            if item.kind[0] == "form":
                for st in item.kind[1].states:
                    for i in st.items:
                        stack.append(i)
                        stack.extend(i.children)
        w._media_uids = out
    return w._media_uids


def _timing_targets(w):
    """UIDs a ``TimingTarget`` can refer to: the page items, movies
    and sounds written and the behaviours of the buttons among the
    items."""
    if w._timing_targets is None:
        out = set(written_media(w))
        for item in w.written_page_items().values():
            out.add(item.uid)
            if item.kind[0] == "form" and item.kind[1].kind == "button":
                out.update(b.uid for b in item.kind[1].behaviors)
        w._timing_targets = out
    return w._timing_targets


def timing_setting(w, x, owner, lists):
    """The ``TimingSetting`` of a spread or button whose ``Self`` is
    ``owner``; nothing without timing lists. A target not written in
    the package is left out, and so are the groups and lists left
    empty."""
    written = _timing_targets(w)
    # Per list: (index, list, kept groups (index, group, kept
    # targets)). `Placement` numbers the kept targets.
    kept = []
    for i, lst in enumerate(lists):
        groups = []
        for j, g in enumerate(lst.groups):
            targets = []
            for t in g.targets:
                if t.uid in written:
                    targets.append(t)
                else:
                    w.warn(f"timing target {t.uid} of {owner!r} is not "
                           f"written; left out")
            if targets:
                groups.append((j, g, targets))
        if groups:
            kept.append((i, lst, groups))
    if not kept:
        return
    x.start("TimingSetting")
    for i, lst, groups in kept:
        list_self = f"{owner}TimingSetting1TimingList{i}"
        x.start("TimingList").attr("Self", list_self)
        trigger = _code(TRIGGERS, lst.trigger)
        if trigger is not None:
            x.attr("TriggerEvent", trigger)
        else:
            w.warn(f"{owner}: trigger code {lst.trigger} is not known; "
                   f"TriggerEvent left out")
        for j, g, targets in groups:
            group_self = f"{list_self}TimingGroup{j}"
            x.start("TimingGroup").attr("Self", group_self)
            if 1 <= g.plays <= 100:
                x.attr("Plays", str(g.plays))
            else:
                w.warn(f"{group_self}: Plays {g.plays} left out")
            x.attr("PlaysLoop", "false")
            for k, t in enumerate(targets):
                target_self = f"{group_self}i{t.uid:x}"
                x.start("TimingTarget") \
                    .attr("Self", target_self) \
                    .attr("DynamicTarget", uref(t.uid))
                if 0.0 <= t.delay <= 60.0:
                    x.attr("DelaySeconds", num(t.delay))
                else:
                    w.warn(f"{target_self}: DelaySeconds {t.delay} left out")
                x.attr("ReverseAnimation", "false") \
                    .attr("TargetRole", str(t.role)) \
                    .attr("TargetAction", str(t.action)) \
                    .attr("Placement", str(k)) \
                    .end()
            x.end()
        x.end()
    x.end()


def animation_setting(w, x, uid, a):
    """The ``AnimationSetting`` of the page item ``uid``."""
    x.start("AnimationSetting").attr(
        "TransformOffsets",
        f"{_six_digits(a.offsets[0])} {_six_digits(a.offsets[1])}")
    if 0.125 <= a.duration <= 60.0:
        x.attr("Duration", num(a.duration))
    else:
        w.warn(f"item {uid}: animation Duration {a.duration} left out")
    design = _code(DESIGN_OPTIONS, a.design)
    if design is not None:
        x.attr("DesignOption", design)
    else:
        w.warn(f"item {uid}: animation design option {a.design} is not "
               f"known; left out")
    ease = _code(EASE_TYPES, a.ease)
    if ease is not None:
        x.attr("EaseType", ease)
    else:
        w.warn(f"item {uid}: animation ease code {a.ease} is not known; "
               f"left out")
    if 1 <= a.plays <= 100:
        x.attr("Plays", str(a.plays))
    else:
        w.warn(f"item {uid}: animation Plays {a.plays} left out")
    x.attr("PlaysLoop", "false") \
        .attr("InitiallyHidden", "false") \
        .attr("HiddenAfter", "false") \
        .attr("HasCustomSettings", str(a.custom).lower())
    x.start("Properties")
    if a.preset != 0:
        x.start("Preset").attr("type", "object").text(uref(a.preset)).end()
    if a.path:
        x.start("MotionPath")
        for frame, px, py in a.path:
            p = _point(px, py)
            x.start("AnimationDataPathKeyFrameType") \
                .attr("KeyFrame", str(frame))
            x.empty("PathPoint", [
                ("Anchor", p),
                ("LeftDirection", p),
                ("RightDirection", p),
            ])
            x.end()
        x.end()
    elif a.points:
        x.start("MotionPathPoints") \
            .attr("PathOpen", "true") \
            .start("PathPointArray")
        for px, py in a.points:
            p = _point(px, py)
            x.empty("PathPointType", [
                ("Anchor", p),
                ("LeftDirection", p),
                ("RightDirection", p),
            ])
        x.end().end()
    for name, tag in PROPERTY_ARRAYS:
        keys = next((k for n, k in a.properties if n == name), None)
        if not keys:
            continue
        x.start(tag)
        for frame, v in keys:
            x.empty("AnimationDataKeyFrameType", [
                ("KeyFrame", str(frame)),
                ("Value", num(v)),
            ])
        x.end()
    x.end().end()
