"""Transparency settings of page items and object styles.

A faithful port of ``src/idml/transparency.rs`` from storytold/indd-utils
(commit 136f973, dual MIT/Apache-2.0 — see attribution.py). The tables
below were translated mechanically by
~/workspace/inddutils-verify/translate_transparency.py; do not edit them
by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

from .idml_attrs import is_code, value_text
from .idml_values import Node
from .idml_xml import num, uref
from .idml_xml import round as _round

BLEND_MODES = ((0, 'Normal'), (1, 'Multiply'), (2, 'Screen'), (3, 'Overlay'), (4, 'SoftLight'), (5, 'HardLight'), (6, 'ColorDodge'), (7, 'ColorBurn'), (8, 'Darken'), (9, 'Lighten'), (10, 'Difference'), (11, 'Exclusion'), (12, 'Hue'), (13, 'Saturation'), (14, 'Color'), (15, 'Luminosity'))

SETTINGS = ["TransparencySetting", "StrokeTransparencySetting", "FillTransparencySetting"]

EFFECTS = ["BlendingSetting", "DropShadowSetting", "FeatherSetting", "InnerShadowSetting", "OuterGlowSetting", "InnerGlowSetting", "BevelAndEmbossSetting", "SatinSetting", "GradientFeatherSetting"]

ATTRS = [
    (67607, 'TransparencySetting', 'BlendingSetting', 'BlendMode', ('enum', 'BLEND_MODES'), 'both'),
    (67606, 'TransparencySetting', 'BlendingSetting', 'Opacity', ('range', 0.0, 100.0), 'both'),
    (67609, 'TransparencySetting', 'BlendingSetting', 'IsolateBlending', ('bool',), 'both'),
    (67610, 'TransparencySetting', 'DropShadowSetting', 'Mode', ('enum', ((0, 'None'), (1, 'Drop'))), 'both'),
    (67615, 'TransparencySetting', 'DropShadowSetting', 'BlendMode', ('enum', 'BLEND_MODES'), 'both'),
    (67614, 'TransparencySetting', 'DropShadowSetting', 'Opacity', ('range', 0.0, 100.0), 'both'),
    (67611, 'TransparencySetting', 'DropShadowSetting', 'XOffset', ('range', -1000.0, 1000.0), 'both'),
    (67612, 'TransparencySetting', 'DropShadowSetting', 'YOffset', ('range', -1000.0, 1000.0), 'both'),
    (67616, 'TransparencySetting', 'DropShadowSetting', 'Size', ('range', 0.0, 1000.0), 'both'),
    (67658, 'TransparencySetting', 'DropShadowSetting', 'Spread', ('range', 0.0, 100.0), 'both'),
    (67639, 'TransparencySetting', 'DropShadowSetting', 'EffectColor', ('swatch',), 'both'),
    (125807, 'TransparencySetting', 'DropShadowSetting', 'KnockedOut', ('bool',), 'both'),
    (67617, 'TransparencySetting', 'FeatherSetting', 'Mode', ('enum', ((0, 'None'), (3, 'Standard'))), 'styles'),
    (67661, 'TransparencySetting', 'InnerShadowSetting', 'Applied', ('bool',), 'both'),
    (67662, 'TransparencySetting', 'InnerShadowSetting', 'EffectColor', ('swatch',), 'both'),
    (67664, 'TransparencySetting', 'InnerShadowSetting', 'Opacity', ('range', 0.0, 100.0), 'styles'),
    (67666, 'TransparencySetting', 'InnerShadowSetting', 'Distance', ('range', 0.0, 1000.0), 'both'),
    (67669, 'TransparencySetting', 'InnerShadowSetting', 'Size', ('range', 0.0, 1000.0), 'both'),
    (67671, 'TransparencySetting', 'OuterGlowSetting', 'Applied', ('bool',), 'both'),
    (67672, 'TransparencySetting', 'OuterGlowSetting', 'BlendMode', ('enum', 'BLEND_MODES'), 'both'),
    (67673, 'TransparencySetting', 'OuterGlowSetting', 'Opacity', ('range', 0.0, 100.0), 'both'),
    (67675, 'TransparencySetting', 'OuterGlowSetting', 'EffectColor', ('swatch',), 'both'),
    (67677, 'TransparencySetting', 'OuterGlowSetting', 'Spread', ('range', 0.0, 100.0), 'both'),
    (67678, 'TransparencySetting', 'OuterGlowSetting', 'Size', ('range', 0.0, 1000.0), 'both'),
    (67679, 'TransparencySetting', 'InnerGlowSetting', 'Applied', ('bool',), 'both'),
    (67680, 'TransparencySetting', 'InnerGlowSetting', 'BlendMode', ('enum', 'BLEND_MODES'), 'both'),
    (67686, 'TransparencySetting', 'InnerGlowSetting', 'Size', ('range', 0.0, 1000.0), 'both'),
    (67701, 'TransparencySetting', 'BevelAndEmbossSetting', 'ShadowColor', ('swatch',), 'styles'),
    (125834, 'TransparencySetting', 'GradientFeatherSetting', 'Applied', ('bool',), 'both'),
    (125837, 'TransparencySetting', 'GradientFeatherSetting', 'Angle', ('range', -180.0, 180.0), 'both'),
    (125838, 'TransparencySetting', 'GradientFeatherSetting', 'Length', ('number',), 'both'),
    (125839, 'TransparencySetting', 'GradientFeatherSetting', 'GradientStart', ('point',), 'both'),
    (125841, 'TransparencySetting', 'GradientFeatherSetting', 'HiliteAngle', ('number',), 'both'),
    (125842, 'TransparencySetting', 'GradientFeatherSetting', 'HiliteLength', ('number',), 'both'),
    (67730, 'StrokeTransparencySetting', 'BlendingSetting', 'BlendMode', ('enum', 'BLEND_MODES'), 'styles'),
    (67759, 'StrokeTransparencySetting', 'OuterGlowSetting', 'Applied', ('bool',), 'styles'),
    (67761, 'StrokeTransparencySetting', 'OuterGlowSetting', 'Opacity', ('range', 0.0, 100.0), 'styles'),
    (67763, 'StrokeTransparencySetting', 'OuterGlowSetting', 'EffectColor', ('swatch',), 'styles'),
    (67765, 'StrokeTransparencySetting', 'OuterGlowSetting', 'Spread', ('range', 0.0, 100.0), 'styles'),
    (125843, 'StrokeTransparencySetting', 'GradientFeatherSetting', 'Applied', ('bool',), 'both'),
    (125846, 'StrokeTransparencySetting', 'GradientFeatherSetting', 'Angle', ('range', -180.0, 180.0), 'both'),
    (125847, 'StrokeTransparencySetting', 'GradientFeatherSetting', 'Length', ('number',), 'both'),
    (125848, 'StrokeTransparencySetting', 'GradientFeatherSetting', 'GradientStart', ('point',), 'both'),
    (67801, 'FillTransparencySetting', 'BlendingSetting', 'BlendMode', ('enum', 'BLEND_MODES'), 'styles'),
    (67800, 'FillTransparencySetting', 'BlendingSetting', 'Opacity', ('range', 0.0, 100.0), 'both'),
    (67804, 'FillTransparencySetting', 'DropShadowSetting', 'Mode', ('enum', ((0, 'None'), (1, 'Drop'))), 'styles'),
    (67805, 'FillTransparencySetting', 'DropShadowSetting', 'XOffset', ('range', -1000.0, 1000.0), 'styles'),
    (67806, 'FillTransparencySetting', 'DropShadowSetting', 'YOffset', ('range', -1000.0, 1000.0), 'styles'),
    (67813, 'FillTransparencySetting', 'DropShadowSetting', 'UseGlobalLight', ('bool',), 'styles'),
    (67820, 'FillTransparencySetting', 'InnerShadowSetting', 'Applied', ('bool',), 'styles'),
    (67821, 'FillTransparencySetting', 'InnerShadowSetting', 'EffectColor', ('swatch',), 'styles'),
    (67823, 'FillTransparencySetting', 'InnerShadowSetting', 'Opacity', ('range', 0.0, 100.0), 'styles'),
    (67824, 'FillTransparencySetting', 'InnerShadowSetting', 'Angle', ('range', -180.0, 180.0), 'styles'),
    (67825, 'FillTransparencySetting', 'InnerShadowSetting', 'Distance', ('range', 0.0, 1000.0), 'styles'),
    (67827, 'FillTransparencySetting', 'InnerShadowSetting', 'ChokeAmount', ('number',), 'styles'),
    (67828, 'FillTransparencySetting', 'InnerShadowSetting', 'Size', ('range', 0.0, 1000.0), 'styles'),
    (67830, 'FillTransparencySetting', 'OuterGlowSetting', 'Applied', ('bool',), 'styles'),
    (67831, 'FillTransparencySetting', 'OuterGlowSetting', 'BlendMode', ('enum', 'BLEND_MODES'), 'styles'),
    (67834, 'FillTransparencySetting', 'OuterGlowSetting', 'EffectColor', ('swatch',), 'styles'),
    (67837, 'FillTransparencySetting', 'OuterGlowSetting', 'Size', ('range', 0.0, 1000.0), 'styles'),
    (67838, 'FillTransparencySetting', 'InnerGlowSetting', 'Applied', ('bool',), 'styles'),
    (67839, 'FillTransparencySetting', 'InnerGlowSetting', 'BlendMode', ('enum', 'BLEND_MODES'), 'styles'),
    (125696, 'FillTransparencySetting', 'InnerGlowSetting', 'Opacity', ('range', 0.0, 100.0), 'styles'),
    (125698, 'FillTransparencySetting', 'InnerGlowSetting', 'EffectColor', ('swatch',), 'styles'),
    (125701, 'FillTransparencySetting', 'InnerGlowSetting', 'Size', ('range', 0.0, 1000.0), 'styles'),
    (125703, 'FillTransparencySetting', 'BevelAndEmbossSetting', 'Applied', ('bool',), 'styles'),
    (125708, 'FillTransparencySetting', 'BevelAndEmbossSetting', 'Size', ('range', 0.0, 1000.0), 'styles'),
    (125709, 'FillTransparencySetting', 'BevelAndEmbossSetting', 'Soften', ('number',), 'styles'),
    (125710, 'FillTransparencySetting', 'BevelAndEmbossSetting', 'Angle', ('range', -180.0, 180.0), 'styles'),
    (125711, 'FillTransparencySetting', 'BevelAndEmbossSetting', 'Altitude', ('number',), 'styles'),
    (125715, 'FillTransparencySetting', 'BevelAndEmbossSetting', 'HighlightOpacity', ('range', 0.0, 100.0), 'styles'),
    (125718, 'FillTransparencySetting', 'BevelAndEmbossSetting', 'ShadowOpacity', ('range', 0.0, 100.0), 'styles'),
    (125719, 'FillTransparencySetting', 'SatinSetting', 'Applied', ('bool',), 'styles'),
    (125720, 'FillTransparencySetting', 'SatinSetting', 'EffectColor', ('swatch',), 'styles'),
    (125721, 'FillTransparencySetting', 'SatinSetting', 'BlendMode', ('enum', 'BLEND_MODES'), 'styles'),
    (125722, 'FillTransparencySetting', 'SatinSetting', 'Opacity', ('range', 0.0, 100.0), 'styles'),
    (125723, 'FillTransparencySetting', 'SatinSetting', 'Angle', ('range', -180.0, 180.0), 'styles'),
    (125724, 'FillTransparencySetting', 'SatinSetting', 'Distance', ('range', 0.0, 1000.0), 'styles'),
    (125725, 'FillTransparencySetting', 'SatinSetting', 'Size', ('range', 0.0, 1000.0), 'styles'),
    (125726, 'FillTransparencySetting', 'SatinSetting', 'InvertEffect', ('bool',), 'styles'),
    (125852, 'FillTransparencySetting', 'GradientFeatherSetting', 'Applied', ('bool',), 'both'),
    (125855, 'FillTransparencySetting', 'GradientFeatherSetting', 'Angle', ('range', -180.0, 180.0), 'both'),
    (125856, 'FillTransparencySetting', 'GradientFeatherSetting', 'Length', ('number',), 'both'),
    (125857, 'FillTransparencySetting', 'GradientFeatherSetting', 'GradientStart', ('point',), 'both'),
]

# The translator kept the Rust table name for the blend-mode enum;
# resolve it to the table itself so `value()` can iterate it.
ATTRS = [
    (a, s, e, n, ("enum", BLEND_MODES) if k == ("enum", "BLEND_MODES") else k, o)
    for a, s, e, n, k, o in ATTRS
]


#: Opacity gradient stops of a gradient feather, per setting.
STOPS = [
    (0x1EB8C, "TransparencySetting"),
    (0x1EB95, "StrokeTransparencySetting"),
    (0x1EB9E, "FillTransparencySetting"),
]

#: The applied flag of each effect whose colour is written only when
#: the effect is applied.
COLOR_APPLIED = [("InnerShadowSetting", 0x1084D)]

#: The drop shadow's ``EffectColor``.
DROP_SHADOW_COLOR = 0x10837

#: The drop shadow's ``KnockedOut``.
KNOCKED_OUT = 0x1EB6F


def _unstored(attr_id):
    """The value of an attribute that an item or style chain does not
    store, where the samples show one."""
    if attr_id == KNOCKED_OUT:
        from .attrs import Value
        return Value(kind="enum", data=1)
    return None


@dataclass
class Stop:
    """One opacity gradient stop, as IDML writes it."""

    opacity: float
    location: float
    #: Relative position of the midpoint between the previous stop and
    #: this one, in percent; ``None`` for the first stop.
    midpoint: float | None = None


def _stops(raw):
    """Opacity stops as IDML writes them, from the stops of the
    attribute list (location, absolute midpoint position, opacity).
    ``None`` for no stops, or two stops at the same location (the
    midpoint is undefined)."""
    if not raw or any(raw[i][0] == raw[i - 1][0] for i in range(1, len(raw))):
        return None
    out = []
    for i, (location, _mid, opacity) in enumerate(raw):
        if i > 0:
            prev, mid, _ = raw[i - 1]
            midpoint = (mid - prev) / (location - prev) * 100.0
        else:
            midpoint = None
        out.append(Stop(opacity=opacity,
                        location=_round(location * 100.0),
                        midpoint=midpoint))
    return out


def _settings(w, attrs, base, owner):
    """The setting elements of ``attrs``, with the values that differ
    from ``base`` in any digit (all values of an item without a style),
    and an effect element only when one of its values or its opacity
    stops differ. ``base`` is ``("item", style_or_None)`` or
    ``("root", root_attrs)``. ``owner`` is the ``Self`` of the item or
    style, which names the opacity stops. Returns (nodes, left_out)
    where left_out is (effect, attribute, value) for values outside
    the schema's range."""
    left_out = []
    out = []
    for setting in SETTINGS:
        effects = []
        for effect in EFFECTS:
            values = []
            for attr_id, s, e, name, kind, on in ATTRS:
                if s != setting or e != effect:
                    continue
                if base[0] == "item":
                    style = base[1]
                    if on == "styles":
                        continue
                    if kind[0] == "swatch":
                        applied = next((a for eff, a in COLOR_APPLIED
                                        if eff == effect), None)
                        if applied is not None and \
                                attrs.get(applied) is not None and \
                                attrs.get(applied).as_u32() != 1:
                            continue
                    v = attrs.get(attr_id)
                    if v is None and style is not None:
                        v = _unstored(attr_id)
                    base_value = None
                    if style is not None:
                        base_value = style.transparency_value(attr_id)
                        if base_value is None:
                            base_value = _unstored(attr_id)
                    has_base = style is not None
                else:
                    root = base[1]
                    v = attrs.get(attr_id)
                    base_value = root.get(attr_id)
                    if v is None or base_value is None:
                        continue
                    has_base = True
                if v is None:
                    continue
                if kind[0] == "range":
                    lo, hi = kind[1], kind[2]
                    f = v.as_f64()
                    if f is not None and not lo <= f <= hi:
                        left_out.append((effect, name, f))
                        continue
                text = value_text(w, kind, v)
                if text is not None:
                    base_text = None
                    if base_value is not None:
                        base_text = value_text(w, kind, base_value)
                    same = base_text is not None and base_text == text
                    unshown = attr_id == DROP_SHADOW_COLOR and has_base \
                        and base_text is None and text == "Color/Black"
                    if not same and not unshown:
                        values.append((name, text))
                elif is_code(kind):
                    attrs.unknown_code(attr_id, v)
            if effect == "GradientFeatherSetting":
                stop_id = next((i for i, s in STOPS if s == setting), None)
                stop_list = []
                if stop_id is not None:
                    if base[0] == "item":
                        style = base[1]
                        base_stops = style.transparency_value(stop_id) \
                            if style is not None else None
                    else:
                        base_stops = base[1].get(stop_id)
                    raw = attrs.get(stop_id)
                    if raw is not None and raw.kind == "stops":
                        same_stops = base_stops is not None \
                            and base_stops.kind == "stops" \
                            and base_stops.data == raw.data
                        if not same_stops:
                            stop_list = _stops(raw.data) or []
            else:
                stop_list = []
            if values or stop_list:
                effects.append((effect, values, stop_list))
        if not effects:
            continue
        children = []
        for effect, values, stop_list in effects:
            stop_nodes = []
            for i, s in enumerate(stop_list):
                sattrs = [
                    ("Self", f"{owner}{setting}1{effect}1OpacityGradientStop{i}"),
                    ("Opacity", num(s.opacity)),
                    ("Location", num(s.location)),
                ]
                if s.midpoint is not None:
                    sattrs.append(("Midpoint", num(s.midpoint)))
                stop_nodes.append(Node(tag="OpacityGradientStop",
                                       attrs=sattrs))
            children.append(Node(tag=effect,
                                 attrs=[(n, v) for n, v in values],
                                 children=stop_nodes))
        out.append(Node(tag=setting, children=children))
    return out, left_out


def write(w, x, attrs, style, item):
    """Write the transparency settings of a page item: the values that
    differ from the object style ``style`` (all values without a
    style). ``item`` is the item's ``Self``. Returns the values left
    out because they are outside the schema's range, as
    (effect, attribute, value)."""
    nodes, left_out = _settings(w, attrs, ("item", style), item)
    for n in nodes:
        n.write(x)
    return left_out


def style_settings(w, os, root):
    """The transparency settings of object style ``os``: the values
    that differ from those of the root style ``[None]``, whatever the
    style is based on. None for an InDesign 4.0 document, whose lists
    have another layout."""
    if w.doc.version.major <= 4:
        return [], []
    return _settings(w, os.transparency, ("root", root.transparency),
                     uref(os.uid))
