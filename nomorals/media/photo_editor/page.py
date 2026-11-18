"""The Studio photo page — one open document as plain HTML.

The API's CSP allows no scripts and no form submissions, so the
page is a VIEW, not a scripted editor: the current composite as a
data-URI PNG, the layer list, the op history, and the notes
(including what the flatten deliberately did not render). Edits
arrive through the chat verb and the brain tool, and the page
re-renders showing them. Every byte is escaped, like the
public-face pages.

A theme (``theme=``) wears the Studio's look: the API's CSP is
``style-src 'none'``, so no stylesheet or style attribute can
land — the palette rides the body's presentational attributes
(``bgcolor``/``text``/``link``/``vlink``), the one styling the CSP
permits. Panels (``panels=``) are declarative sections contributed
by enabled Studio plug-ins, rendered escaped in their slots.
"""

from __future__ import annotations

import base64
import html as _html
from typing import Any

__all__ = ["photo_page_html"]


def photo_page_html(doc: Any, *, doc_id: str = "", theme: Any = None,
                    theme_name: str = "", panels: Any = ()) -> str:
    """Render an open photo document as a self-contained page.

    ``theme`` is a Studio palette (see :mod:`nomorals.studio`):
    its colours ride the body's presentational attributes, the
    only styling the API CSP permits. ``panels`` are parsed
    ``panel.json`` files from enabled Studio plug-ins, rendered
    in their slots (``above`` first, ``below`` last).
    """
    png = base64.b64encode(doc.render_png_bytes()).decode("ascii")
    title = _html.escape(doc.title or "untitled")
    ident = _html.escape(doc_id or "")
    body_attrs = ""
    if theme is not None:
        body_attrs = (
            f" bgcolor=\"{_html.escape(str(theme.background))}\""
            f" text=\"{_html.escape(str(theme.text))}\""
            f" link=\"{_html.escape(str(theme.accent))}\""
            f" vlink=\"{_html.escape(str(theme.muted_text))}\"")
    lines = [
        "<!DOCTYPE html>",
        "<html lang=\"en\"><head><meta charset=\"utf-8\">",
        f"<title>Photo — {title}</title></head><body{body_attrs}>",
        f"<h1>Photo — {title}</h1>",
    ]
    if theme is not None and theme_name:
        lines.append(f"<p>Theme: {_html.escape(theme_name)}</p>")
    for panel in panels or ():
        if getattr(panel, "slot", "") == "above":
            lines.extend(_panel_lines(panel))
    if ident:
        lines.append(f"<p>Document <code>{ident}</code></p>")
    lines.append(
        f"<p>{doc.width}×{doc.height}, {len(doc.layers)} layers"
        "</p>")
    lines.append(
        "<figure><img alt=\"current composite\" "
        f"src=\"data:image/png;base64,{png}\">"
        "<figcaption>Current composite</figcaption></figure>")
    lines.append("<h2>Layers</h2><ol>")
    for i, layer in enumerate(doc.layers):
        name = _html.escape(layer.name)
        kind = _html.escape(layer.kind)
        lines.append(
            f"<li>{i}. {name} — {kind} "
            f"{layer.width}×{layer.height} at "
            f"({layer.left},{layer.top}) "
            f"opacity {layer.opacity}"
            + ("" if layer.visible else " (hidden)")
            + "</li>")
    lines.append("</ol>")
    if doc.history:
        lines.append("<h2>Edits</h2><ol>")
        for op in doc.history:
            params = ", ".join(
                f"{_html.escape(str(k))}="
                f"{_html.escape(str(v))}"
                for k, v in op["params"].items())
            lines.append(
                f"<li>{_html.escape(op['op'])}({params})</li>")
        lines.append("</ol>")
    if doc.notes:
        lines.append("<h2>Notes</h2><ul>")
        for note in doc.notes:
            lines.append(f"<li>{_html.escape(note)}</li>")
        lines.append("</ul>")
    for panel in panels or ():
        if getattr(panel, "slot", "") != "above":
            lines.extend(_panel_lines(panel))
    lines.append(
        "<p>Edit it in chat: "
        "<code>/photo edit &lt;id&gt; crop 0 0 128 128</code> · "
        "<code>/photo edit &lt;id&gt; enhance brightness 1.2</code> · "
        "<code>/photo save &lt;id&gt; &lt;path&gt;</code></p>")
    lines.append("</body></html>")
    return "\n".join(lines)


def _panel_lines(panel: Any) -> list:
    """One plug-in panel as escaped HTML lines."""
    lines = [f"<section><h2>{_html.escape(panel.title)}</h2>"]
    for block in getattr(panel, "blocks", ()):
        kind = block.get("kind", "")
        if kind == "text":
            lines.append(f"<p>{_html.escape(block.get('text', ''))}</p>")
        elif kind == "kv":
            lines.append(
                "<dl><dt>"
                f"{_html.escape(block.get('label', ''))}</dt><dd>"
                f"{_html.escape(block.get('value', ''))}</dd></dl>")
        elif kind == "link":
            lines.append(
                "<p><a href=\""
                f"{_html.escape(block.get('href', ''), quote=True)}\">"
                f"{_html.escape(block.get('label', ''))}</a></p>")
    lines.append("</section>")
    return lines
