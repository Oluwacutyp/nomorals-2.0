"""Pass #217 pins: books/library.py — 1 exclusion, 0 converts.

- 812: ``Library._corpus_version`` store-read failure. The corpus
  version stamp is a freshness signal for the cached search index.
  When the store read itself fails, the answer is 0 ("unversioned")
  — and 0 is the SAFE direction: the next search sees the new stamp
  (or no stamp) and rebuilds its index instead of answering from a
  stale one. The swallow is the correct behavior: a failed freshness
  read degrades to "rebuild", and the function's contract already
  produces 0 for "unversioned" on a non-error path.

books/library.py DOES call record_failure elsewhere
(library._strip_html_fragment, library._bytes_to_text), so
NOT-recording is pinned against the books ledger: books
recent_failures() before/after each trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use a real Library with a
failing store, never a swallow.
"""

import tempfile
from types import SimpleNamespace

from nomorals.books import failures as books_fail
from nomorals.books.library import Library


def _snap():
    return list(books_fail.recent_failures(limit=200))


def _lib():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    ctx = SimpleNamespace(db=None, settings=settings)
    return Library(ctx)


class _BoomStore:
    def get(self, *args, **kwargs):
        raise OSError("store read boom")


# ── L812: a failed freshness read degrades to 0 (rebuild) ──────────

def test_812_store_read_failure_returns_zero():
    lib = _lib()
    lib._port = SimpleNamespace(store=_BoomStore(),
                                meta_collection="meta")
    before = _snap()
    assert lib._corpus_version() == 0
    assert _snap() == before


def test_812_no_port_returns_zero():
    lib = _lib()
    lib._port = None
    before = _snap()
    assert lib._corpus_version() == 0
    assert _snap() == before


def test_812_unparseable_version_returns_zero():
    class _BadStore:
        def get(self, *args, **kwargs):
            return {"v": "not-a-number"}

    lib = _lib()
    lib._port = SimpleNamespace(store=_BadStore(),
                                meta_collection="meta")
    before = _snap()
    assert lib._corpus_version() == 0
    assert _snap() == before


def test_812_good_version_is_returned():
    class _GoodStore:
        def get(self, *args, **kwargs):
            return {"v": 7}

    lib = _lib()
    lib._port = SimpleNamespace(store=_GoodStore(),
                                meta_collection="meta")
    before = _snap()
    assert lib._corpus_version() == 7
    assert _snap() == before
