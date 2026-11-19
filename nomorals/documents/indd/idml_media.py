"""Placed movies and sounds (``Movie``, ``Sound``) with their poster
and link.

A faithful port of ``src/idml/media.rs`` from storytold/indd-utils
(commit 136f973, dual MIT/Apache-2.0 — see attribution.py).
"""

from __future__ import annotations

from .idml_spread import label, path_geometry_element, text_wrap_preference
from .idml_transparency import write as transparency_write
from .idml_xml import matrix, num, uref
from .model import percent_decode


def _native_path(uri):
    """``FilePath`` of a media link: the URI without ``file:``,
    percent-decoded, with backslashes in a path that starts with a
    drive letter."""
    path = percent_decode(uri[5:] if uri.startswith("file:") else uri)
    if len(path) >= 3 and path[0].isalpha() and path[1] == ":" \
            and path[2] == "/":
        return path.replace("/", "\\")
    return path


def media(w, x, m, outer):
    """A ``Movie`` or ``Sound`` element. ``outer`` is the transform
    from the media's frame to the spread."""
    from .idml_spread import settings as _settings  # Writer.settings

    s = m.settings
    x.start("Movie" if m.kind == "movie" else "Sound") \
        .attr("Self", uref(m.uid))
    if s.name is not None:
        x.attr("Name", s.name)
    x.attr("Description", s.description)
    file_path = _native_path(m.link.uri) if m.link is not None else None
    if m.kind == "movie":
        controller = w.dom_version() < (18, 0)
        x.attr("FloatingWindowPosition", "Center") \
            .attr("FloatingWindowSize", "Full") \
            .attr("PlayOnPageTurn", str(s.play_on_page_turn).lower())
        if controller:
            x.attr("ShowControls", "false")
        x.attr("FloatingWindow", "false") \
            .attr("EmbedInPDF", str(s.embed_in_pdf).lower()) \
            .attr("PlayMode", "RepeatPlay" if s.looped else "Once") \
            .attr("CustomPoster", str(s.custom_poster).lower())
        if file_path is not None:
            x.attr("FilePath", file_path)
        if controller:
            x.attr("ControllerSkin", s.controller_skin) \
                .attr("ShowController", str(s.show_controller).lower())
        x.attr("MovieLoop", str(s.looped).lower())
    else:
        x.attr("PlayOnPageTurn", str(s.play_on_page_turn).lower()) \
            .attr("DoNotPrintPoster", "false") \
            .attr("EmbedInPDF", str(s.embed_in_pdf).lower())
        if file_path is not None:
            x.attr("FilePath", file_path)
        x.attr("StopOnPageTurn", str(s.stop_on_page_turn).lower()) \
            .attr("SoundLoop", str(s.looped).lower())
    x.attr("ItemTransform", matrix(m.transform))
    # The name comes from the playback settings, not chunk 0x2C10.
    from .model import ItemName, ItemProps
    props = ItemProps(name=ItemName(False, s.name) if s.name is not None else None,
                      hidden=m.props.hidden, locked=m.props.locked,
                      allow_overrides=None,
                      change_counts=m.props.change_counts,
                      overridden=m.props.overridden,
                      layout_constraints=m.props.layout_constraints)
    _settings(w, x, props, None)
    if s.name is None:
        x.remove_attr("Name")
    x.attr("LocalDisplaySetting", "Default")
    if m.object_style is not None:
        ref = w.object_style_ref(m.object_style)
        if ref is not None:
            x.attr("AppliedObjectStyle", ref)
    left, top, right, bottom = m.bounds
    x.start("Properties")
    if m.paths:
        path_geometry_element(x, m.paths)
    x.empty("GraphicBounds", [
        ("Left", num(left)),
        ("Top", num(top)),
        ("Right", num(right)),
        ("Bottom", num(bottom)),
    ])
    label(x, m.label)
    x.end()
    from .idml_applied import style_values
    applied = style_values(w, m.object_style) \
        if m.object_style is not None else None
    for effect, attr, v in transparency_write(w, x, m.attrs, applied,
                                              uref(m.uid)):
        w.warn(f"media {m.uid}: {effect} {attr} {v} is outside the "
               f"IDML range; left out")
    # The contour of the poster, as for a frame and its graphic.
    from .idml_graphic import graphic_contour
    contour = graphic_contour(w, m.poster) if m.poster is not None else None
    text_wrap_preference(x, m.text_wrap, contour)
    if m.poster is not None:
        from .idml_graphic import placed_graphic
        placed_graphic(w, x, m.poster, m.transform.then(outer))
    if m.link is not None:
        from .idml_graphic import link
        link(w, x, m.link)
    x.end()
