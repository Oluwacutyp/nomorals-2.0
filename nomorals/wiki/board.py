"""Wiki → board: a page's sections as board nodes, chained in order.

One text node per ``## `` section (title node first), arrows chaining
them in reading order — the page becomes a knowledge map the owner
can see on the board. Built on the board organ's own seams
(``BoardDocument.add_shape`` + ``bind``), never a parallel board.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

__all__ = ["export_to_board"]


def _sections(body: str) -> list[tuple[str, str]]:
    """(heading, section text) pairs; the intro is headed ``""``."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in (body or "").splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
            continue
        sections[-1][1].append(line)
    return [(head, "\n".join(lines).strip()) for head, lines in sections]


def export_to_board(store: Any, slug: str) -> Any:
    """A page as a ``BoardDocument`` titled ``Wiki: <title>``."""
    from ..media.edit_engine.board import BoardDocument

    page = store.get(slug)
    if page is None:
        raise ValueError(f"wiki: no page {slug!r}")
    doc = BoardDocument.create(f"Wiki: {page.title}")
    sections = _sections(page.body)
    node_ids: list[str] = []
    y = 40.0
    for index, (heading, text) in enumerate(sections):
        node_id = f"wiki-{page.slug}-{index}"
        label = heading if heading else page.title
        body = text[:400]
        shape_text = label if not body else f"{label}\n{body}"
        doc.add_shape({
            "id": node_id, "type": "text",
            "x": 80.0, "y": y, "w": 320.0, "h": 110.0,
            "props": {"text": shape_text},
        })
        node_ids.append(node_id)
        y += 150.0
    for src, dst in pairwise(node_ids):
        doc.bind(src, dst, label="")
    return doc
