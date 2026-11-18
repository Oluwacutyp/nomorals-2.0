"""The Studio photo page — one open document as plain HTML.

The API's CSP allows no scripts and no form submissions, so the
page is a VIEW, not a scripted editor: the current composite as a
data-URI PNG, the layer list, the op history, and the notes
(including what the flatten deliberately did not render). Edits
arrive through the chat verb and the brain tool, and the page
re-renders showing them. Every byte is escaped, like the
public-face pages.
"""

from __future__ import annotations

import base64
import html as _html
from typing import Any

__all__ = ["photo_page_html"]


def photo_page_html(doc: Any, *, doc_id: str = "") -> str:
    """Render an open photo document as a self-contained page."""
    png = base64.b64encode(doc.render_png_bytes()).decode("ascii")
    title = _html.escape(doc.title or "untitled")
    ident = _html.escape(doc_id or "")
    lines = [
        "<!DOCTYPE html>",
        "<html lang=\"en\"><head><meta charset=\"utf-8\">",
        f"<title>Photo — {title}</title></head><body>",
        f"<h1>Photo — {title}</h1>",
    ]
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
    lines.append(
        "<p>Edit it in chat: "
        "<code>/photo edit &lt;id&gt; crop 0 0 128 128</code> · "
        "<code>/photo edit &lt;id&gt; enhance brightness 1.2</code> · "
        "<code>/photo save &lt;id&gt; &lt;path&gt;</code></p>")
    lines.append("</body></html>")
    return "\n".join(lines)
