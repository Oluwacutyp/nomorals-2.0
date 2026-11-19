"""Placed graphics (images, PDF, EPS, SVG) with their settings,
clipping path, import options, layers and links.

A faithful port of ``src/idml/graphic.rs`` from storytold/indd-utils
(commit 136f973, dual MIT/Apache-2.0 — see attribution.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from .idml_attrs import gradient_values, item_attr_values
from .idml_spread import geometry_paths, text_wrap_preference
from .idml_spread import settings as item_settings
from .idml_xml import base64_lines, builtin_key, link_time, matrix, num, nums, uref
from .model import ClippingPath


@dataclass
class ContourOption:
    """The ``ContourOption`` of a text wrap."""

    #: Type and inside edges; ``None`` for a frame whose object style
    #: turns the text wrap category on, which has only
    #: ``ContourPathName``.
    contour: object = None
    #: ``ContourPathName``: ``$ID/`` or the name of an alpha channel.
    path_name: str = "$ID/"
    #: Whether to write ``CONTOUR_PARAMS`` with a contour type (DOM
    #: 21.4 on).
    params: bool = False


def _color_space(code, indexed):
    """``Space`` of an image by its colour space code (and whether it
    has a colour table)."""
    if code == 1:
        return "$ID/#Links_Grayscale"
    if code == 2:
        return "$ID/#Links_Indexed RGB" if indexed else "$ID/#Links_RGB"
    if code == 4:
        return "$ID/#Links_CMYK"
    return None


def _profile(code):
    """``Properties/Profile`` of an image by the code of chunk
    0x7C0F."""
    if code is None or code == 0:
        return "$ID/None"
    if code == 1:
        return "$ID/Use Document Default"
    if code == 3:
        return "$ID/Embedded"
    return None


def _vector_policy(code):
    """A vector colour policy code of chunk 0x7C42."""
    if code == 1:
        return "IgnoreAll"
    if code == 3:
        return "HonorAllProfiles"
    return None


def _pdf_crop(code):
    """``PDFCrop`` by the crop code of chunk 0x251B."""
    return {0: "CropContentVisibleLayers", 2: "CropArt", 3: "CropPDF",
            4: "CropTrim", 6: "CropMedia",
            7: "CropContentAllLayers"}.get(code)


def _round_half_up(v):
    """A value rounded half up, as IDML rounds resolutions."""
    import math
    return math.floor(v + 0.5)


def _scales(m):
    """The horizontal and vertical scale of a transform: √(a² + b²),
    and |ad − bc| divided by the horizontal scale."""
    import math
    a, b, c, d = m.values[0], m.values[1], m.values[2], m.values[3]
    h = math.hypot(a, b)
    return h, abs(a * d - b * c) / h


def clipping(x, g):
    """``ClippingPathSettings`` of an image, PDF, EPS or placed page,
    from chunk 0x2C1A; a graphic without the chunk has the values
    every such graphic has in IDML. A value that the stored fields do
    not settle is left out."""
    default = ClippingPath(kind=0, tolerance=2.0, inset=0.0, threshold=25,
                           index=-1, flags=2)
    c = g.clipping if g.clipping is not None else default
    kind = {0: "None", 1: "DetectEdges", 2: "AlphaChannel",
            3: "PhotoshopPath"}.get(c.kind)
    if kind is None:
        return
    if c.kind in (0, 1):
        path_name = "$ID/"
    elif c.kind == 2:
        path_name = g.alpha_channel
    else:
        path_name = g.image.clipping_path_name \
            if g.image is not None else None
    # Bits 0x01 and 0x08 are invert and restrict to frame, in an order
    # not shown: only both set or both clear are read.
    bits = c.flags & 0x09
    invert_restrict = {0: "false", 0x09: "true"}.get(bits)
    x.start("ClippingPathSettings") \
        .attr("UseHighResolutionImage", str(bool(c.flags & 0x02)).lower()) \
        .attr("Threshold", str(c.threshold)) \
        .attr("Tolerance", num(c.tolerance)) \
        .attr("InsetFrame", num(c.inset))
    # The i16 is not the index of an alpha channel.
    if c.kind != 2:
        x.attr("Index", str(c.index))
    x.attr("ClippingType", kind)
    if invert_restrict is not None:
        x.attr("InvertPath", invert_restrict)
    x.attr("IncludeInsideEdges", "false")
    if invert_restrict is not None:
        x.attr("RestrictToFrame", invert_restrict)
    if path_name is not None:
        x.attr("AppliedPathName", path_name)
    x.end()


def _image_io(x, g):
    """``ImageIOPreference`` of an image, from chunk 0x1714; without
    the chunk, the values every image without it has."""
    if g.import_options is not None:
        clipping_flag, alpha = g.import_options
        alpha = alpha.idml()
    else:
        clipping_flag, alpha = True, "$ID/"
    x.empty("ImageIOPreference", [
        ("ApplyPhotoshopClippingPath", str(clipping_flag).lower()),
        ("AllowAutoEmbedding", "true"),
        ("AlphaChannelName", alpha),
    ])


def link(w, x, link_obj):
    """A ``Link`` element. Values without an INDD field are those of
    every link in the corpus IDML files."""
    major = w.doc.version.major
    x.start("Link").attr("Self", uref(link_obj.uid))
    # The values every link has from DOM 21.5.
    dom = w.dom_version()
    if dom >= (21, 5):
        x.attr("AiGeneratedAsset", "false") \
            .attr("AiFullyGeneratedAsset", "false") \
            .attr("AiC2paArchiveData", "$ID/")
    x.attr("AssetURL", "$ID/") \
        .attr("AssetID", "$ID/") \
        .attr("LinkResourceURI", link_obj.uri)
    if link_obj.format is not None:
        x.attr("LinkResourceFormat", builtin_key(link_obj.format))
    if link_obj.embedded:
        state = "Embedded"
    elif link_obj.contained:
        state = "Contained"
    else:
        state = "Normal"
    # The class of the link object.
    klass = link_obj.class_id if link_obj.class_id != 0 else 0x8C42
    x.attr("StoredState", state) \
        .attr("LinkClassID", str(klass)) \
        .attr("LinkClientID", "257")
    if link_obj.modified is not None:
        x.attr("LinkResourceModified", str(link_obj.modified).lower())
    x.attr("LinkObjectModified", "false")
    if link_obj.shown is not None:
        x.attr("ShowInUI", str(link_obj.shown).lower())
    x.attr("CanEmbed", "true") \
        .attr("CanUnembed", "true") \
        .attr("CanPackage", "true") \
        .attr("ImportPolicy", "NoAutoImport") \
        .attr("ExportPolicy", "NoAutoExport")
    if link_obj.stamp is not None:
        x.attr("LinkImportStamp", link_obj.stamp)
    if link_obj.times is not None:
        modified, imported = link_obj.times
        if link_obj.stamp is not None:
            x.attr("LinkImportModificationTime",
                   link_time(modified, w.doc.xmp_dates))
        x.attr("LinkImportTime", link_time(imported, w.doc.xmp_dates))
    if link_obj.size is not None:
        high, low = link_obj.size
        x.attr("LinkResourceSize", f"{high:x}~{low:x}")
    if major >= 13:
        x.attr("RenditionData", "Actual")
    if major >= 21:
        x.attr("PDFIdentifier",
               str(link_obj.pdf_identifier
                   if link_obj.pdf_identifier is not None else 0))
    # Every link has it from DOM 21.4.
    if dom >= (21, 4):
        x.attr("SourceObjectName", "")
    x.end()


def _vector_policies(w, g):
    """The vector colour policies of a PDF or EPS: from chunk 0x7C42,
    or without it from the document's colour policies."""
    def document(name):
        for v in w.doc.prefs.values:
            if v.element == "Document" and v.name == name:
                return v.value
        return None

    rgb = None
    doc_rgb = document("RGBPolicy")
    if doc_rgb == "ColorPolicyOff":
        rgb = "IgnoreAll"
    elif doc_rgb is not None:
        rgb = "HonorAllProfiles"
    cmyk = None
    doc_cmyk = document("CMYKPolicy")
    if doc_cmyk in ("ColorPolicyOff", "CombinationOfPreserveAndSafeCmyk"):
        cmyk = "IgnoreAll"
    elif doc_cmyk in ("PreserveEmbeddedProfiles", "ConvertToWorkingSpace"):
        cmyk = "HonorAllProfiles"
    if g.vector_policies is not None:
        _a, r, _b, c = g.vector_policies
        rgb = _vector_policy(r) or rgb
        cmyk = _vector_policy(c) or cmyk
    return rgb, cmyk


def graphic_contour(w, g):
    """The contour of a graphic's text wrap: its own (chunk 0x373D),
    else that of new graphics in the preferences."""
    c = g.contour if g.contour is not None else w.doc.prefs.contour
    if c is None:
        return None
    # An `AlphaChannel` contour names the image's first channel.
    path_name = g.alpha_channel \
        if g.alpha_channel is not None and c.kind == 2 else "$ID/"
    return ContourOption(contour=c, path_name=path_name,
                         params=w.dom_version() >= (21, 4))


def placed_graphic(w, x, g, outer):
    """A placed graphic. ``outer`` is the transform from the graphic's
    parent to the spread."""
    from .idml_spread import object_style_ref

    major = w.doc.version.major
    tag = {"image": "Image", "pdf": "PDF", "eps": "EPS", "svg": "SVG",
           "imported_page": "ImportedPage"}[g.kind]
    left, top, right, bottom = g.bounds
    x.start(tag).attr("Self", uref(g.uid))
    # The page number is the stored index + 1.
    if g.page_index is not None:
        x.attr("PageNumber", str(g.page_index + 1))
    if g.kind == "image":
        if g.link is not None and g.image is not None:
            p = g.image
            if p.color_space is not None:
                space = _color_space(p.color_space, p.indexed)
                if space is not None:
                    x.attr("Space", space)
            if p.resolution is not None:
                h, v = p.resolution
                actual = [_round_half_up(h), _round_half_up(v)]
                x.attr("ActualPpi", nums(actual))
                sh, sv = _scales(g.transform.then(outer))
                if sh > 0.0 and sv > 0.0:
                    effective = [_round_half_up(actual[0] / sh),
                                 _round_half_up(actual[1] / sv)]
                    import math
                    if all(math.isfinite(e) for e in effective):
                        x.attr("EffectivePpi", nums(effective))
        x.attr("ImageRenderingIntent", "UseColorSettings")
    elif g.kind in ("pdf", "eps"):
        x.attr("GrayVectorPolicy", "IgnoreAll")
        rgb, cmyk = _vector_policies(w, g)
        if rgb is not None:
            x.attr("RGBVectorPolicy", rgb)
        if cmyk is not None:
            x.attr("CMYKVectorPolicy", cmyk)
    elif g.kind == "svg":
        x.attr("UseSVGAs", "EmbedCode")
    x.attr("LocalDisplaySetting", "Default")
    if g.link is not None and g.link.format is not None:
        x.attr("ImageTypeName", builtin_key(g.link.format))
    if g.object_style is not None:
        style = object_style_ref(w, g.object_style)
        if style is not None:
            x.attr("AppliedObjectStyle", style)
    x.attr("ItemTransform", matrix(g.transform))
    # From 21.4; -1 without the chunk.
    if w.saved_by((21, 4)):
        if g.props.grouping_layer is not None:
            x.attr("BeforeGroupingLayerReference",
                   uref(g.props.grouping_layer))
        x.attr("BeforeGroupingLayerPosition",
               str(g.props.grouping_layer_position
                   if g.props.grouping_layer_position is not None else -1))
    # IDML has no `AllowOverrides` on placed graphics.
    from dataclasses import replace
    props = replace(g.props, allow_overrides=None)
    item_settings(w, x, props, None)
    if g.kind == "image":
        for name, v in item_attr_values(w, g.attrs):
            if name == "FillColor":
                x.attr(name, v)
        for name, v in gradient_values(g.attrs):
            if name.startswith("GradientFill"):
                x.attr(name, v)
    if major >= 21 and g.kind not in ("eps", "imported_page"):
        x.attr("FlexItemWidthMode", "FlexFixed") \
            .attr("FlexItemHeightMode", "FlexFixed")
    if g.kind == "svg" and (major, w.doc.version.minor) >= (21, 2):
        x.attr("IsMathMLObject", "false")
    x.start("Properties")
    if g.kind == "image":
        p = _profile(g.profile)
        if p is not None:
            x.start("Profile").attr("type", "string").text(p).end()
    if g.contents is not None:
        x.start("Contents") \
            .cdata(base64_lines(g.contents), 262144) \
            .end()
    if g.clipping_paths:
        geometry_paths(x, "ClippingPathGeometry", g.clipping_paths)
    x.empty("GraphicBounds", [
        ("Left", num(left)),
        ("Top", num(top)),
        ("Right", num(right)),
        ("Bottom", num(bottom)),
    ])
    x.end()
    if g.kind != "svg":
        clipping(x, g)
    if g.kind == "image":
        _image_io(x, g)
    if g.kind == "pdf" or (g.kind == "image" and g.layers is not None
                           and g.layers.option):
        _graphic_layers(x, g)
    text_wrap_preference(x, g.text_wrap, graphic_contour(w, g))
    if g.link is not None:
        link(w, x, g.link)
    if g.kind == "pdf":
        if g.pdf is not None:
            page, transparent, crop = g.pdf
            page = str(page)
            transparent = {0: "false", 1: "true"}.get(transparent)
            crop = _pdf_crop(crop)
        else:
            page, transparent, crop = "1", "true", \
                "CropContentVisibleLayers"
        x.start("PDFAttribute").attr("PageNumber", page)
        if crop is not None:
            x.attr("PDFCrop", crop)
        if transparent is not None:
            x.attr("TransparentBackground", transparent)
        x.end()
    x.end()


def _graphic_layers(x, g):
    """``GraphicLayerOption`` with the graphic's layers, and for an
    image ``LayerCompOption``."""
    layers = g.layers.layers if g.layers is not None else []
    x.start("GraphicLayerOption") \
        .attr("UpdateLinkOption", "KeepOverrides")
    base = f"{uref(g.uid)}GraphicLayerOption1"
    # Top-level layers, then each layer's children, in record order. A
    # layer whose parent is not listed is left out.
    path = []
    _layer_children(x, layers, -1, base, path)
    x.end()
    if g.kind == "image" and g.layer_comp is not None:
        x.empty("LayerCompOption",
                [("AppliedLayerComp", str(g.layer_comp))])


def _layer_children(x, layers, parent, base, path):
    for layer in [layer for layer in layers if layer.parent == parent]:
        # A layer that contains itself would never end.
        if layer.id in path or not -(2 ** 31) <= layer.id < 2 ** 31:
            continue
        path.append(layer.id)
        ident = "".join(f"i{i:x}" for i in path)
        x.start("GraphicLayer") \
            .attr("Self", f"{base}{ident}") \
            .attr("Name", layer.name.idml()) \
            .attr("OriginalVisibility",
                  str(layer.original_visibility).lower()) \
            .attr("CurrentVisibility",
                  str(layer.current_visibility).lower()) \
            .attr("SeparatorLayer", str(bool(layer.flags & 0x01)).lower()) \
            .attr("AdjustmentLayer", "false") \
            .attr("FXLayer", str(bool(layer.flags & 0x04)).lower()) \
            .attr("Locked", str(bool(layer.flags & 0x08)).lower())
        for name in ["HasViewState", "ViewState", "HasExportState",
                     "ExportState", "HasPrintState", "PrintState"]:
            x.attr(name, "false")
        x.attr("Id", str(layer.id))
        _layer_children(x, layers, layer.id, base, path)
        x.end()
        path.pop()
