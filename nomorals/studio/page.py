"""The Studio extensions board — installed extensions as plain HTML.

The API's CSP allows no scripts and no form submissions, so the
board is a VIEW, not a manager: what is installed, what is switched
on for the Studio, and which theme the Studio is wearing. Changes
arrive through the chat verb and the brain tool, and the board
re-renders showing them. Every byte is escaped, like the
public-face pages.
"""

from __future__ import annotations

import html as _html
from typing import Any

__all__ = ["extensions_board_html"]


def extensions_board_html(store: Any) -> str:
    """Render the installed extensions as a self-contained page."""
    state = store.state()
    active = store.active_theme("studio")
    lines = [
        "<!DOCTYPE html>",
        "<html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<title>Studio — Extensions</title></head><body>",
        "<h1>Studio — Extensions</h1>",
    ]
    if active is not None:
        lines.append(
            "<p>Theme: "
            f"{_html.escape(active.name)} "
            f"({_html.escape(active.id)}) — "
            f"background {_html.escape(str(active.palette.background))}"
            "</p>")
    else:
        lines.append("<p>Theme: the Studio's own look</p>")
    if not state.installed:
        lines.append("<p>No extensions installed yet.</p>")
    else:
        lines.append("<h2>Installed</h2><ul>")
        for item in state.installed:
            on = state.is_enabled("studio", str(item.id))
            lines.append(
                f"<li>{_html.escape(str(item.id))} "
                f"{_html.escape(str(item.version))} — "
                f"{_html.escape(item.kind)} — "
                f"{_html.escape(item.name)} "
                f"({_html.escape(item.tier)})"
                + (" — on" if on else " — off")
                + "</li>")
        lines.append("</ul>")
    lines.append(
        "<p>Manage them in chat: "
        "<code>/extensions</code> · "
        "<code>/extensions install &lt;path-or-url&gt;</code> · "
        "<code>/extensions theme &lt;id&gt;</code></p>")
    lines.append("</body></html>")
    return "\n".join(lines)
