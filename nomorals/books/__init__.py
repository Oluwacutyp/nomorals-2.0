"""BookForge: plan, write, build, and send real books.

A book is a durable, resumable object on disk (workspace/books/<slug>/).
The pipeline: create (topic → outline, with live research notes) → write
chapters (model when a real model is answering, template composer when it
is not — never stub text) → build (manuscript.md + a real PDF with cover
title, table of contents with true page numbers, chapters on fresh pages)
→ send the PDF to any live chat platform via the gateway ("send when done").
"""

from .forge import BookForge
from .model import Book, Chapter

__all__ = ["BookForge", "Book", "Chapter"]
