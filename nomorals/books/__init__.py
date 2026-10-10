"""BookForge: plan, write, build, and send real books.

A book is a durable, resumable object on disk (workspace/books/<slug>/).
The pipeline: create (topic → outline, with live research notes) → write
chapters (model when a real model is answering, template composer when it
is not — never stub text) → build (manuscript.md + a real PDF with cover
title, table of contents with true page numbers, chapters on fresh pages)
→ send the PDF to any live chat platform via the gateway ("send when done").

The expansion (Phase 2 §10):

* ``sources`` — native webnovel source adapters (freewebnovel, novelfull,
  royalroad, pandanovel/generic) with a plain-HTTP → browser-context
  fetch chain;
* ``reader`` — the story reader: follow stories, reading progress,
  bookmarks, chapter cache;
* ``bible`` — story bibles auto-built from read chapters (cast, threads,
  world rules, voice);
* ``continuation`` — continue a read story in its own voice;
* ``fiction`` — FictionWriter: genre engines with real mechanics
  (mystery/thriller/horror/sci-fi/fantasy/romance), novel + never-ending
  serial modes, WisdomKeeper thematic weave.
"""

from .forge import BookForge
from .model import Book, Chapter

__all__ = ["BookForge", "Book", "Chapter"]
